"""Host-only, fixed-point evaluation of densely spaced harmonic channel modes.

LABEL: ``eq: smooth channel kernel`` and ``extra eq: channel kernel``;
``[extension]`` denotes the numerical range evaluator. Frequencies have
units Hz, ``B`` Gauss, and powers erg/s/sr per electron. Output shapes are
``I,V: (n_ch,)`` and ``P: (n_weights,n_ch)``.

This is a provisional value and first-derivative backend for the vacuum
helical-orbit reference. ``build_high_order_basis`` uses it in an eager
preparation step; the resulting matrix can then be reused by JIT prediction.
It keeps the finite-gamma line powers and evaluates each line's response and
Faraday weight. It does *not* certify the accelerated result: the special-
function, Euler--Maclaurin remainder, and floating-point errors have no
computed upper bound. It is intentionally separate from ``HarmonicKernel`` and
the low-order ``build_basis`` route. ``build_high_order_basis`` can use its
values while marking uncaught numerical errors ``unbounded``.

The accelerated branch uses the integral and half-endpoint terms of the
Euler--Maclaurin identity, `DLMF 2.10.1 <https://dlmf.nist.gov/2.10.E1>`_.
The identity's remainder contains a weighted integral of ``f''``; a
fine-minus-coarse quadrature difference is *not* a bound on that remainder.
SciPy's ``jv`` evaluates Bessel functions at real continuous order
for the integral. No truncated DLMF Airy expansion is used by this module.

SciPy is optional; install ``syncmoments[high_harmonic]`` to use this module.
The host evaluator itself has no JAX JIT or automatic differentiation. Outputs
are NumPy arrays in the :class:`HostModes` natural-basis layout.
Not certified: the absolute numerical error of the accelerated or direct
result and the physical adequacy of the helical-orbit reference model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from typing import NamedTuple

import numpy as np

from ..constants import C_CGS, C_SI_M, E_ESU, M_E
from ._channel_shapes import FLOOR, unit_peak_area
from ._kernel_helpers import phase_coordinate, phase_weights
from .channels import Channels
from .errors import ErrorTerm
from .phase import TaylorPhase

_MAX_EXACT_FLOAT_INT = 2**53 - 2
_SMOOTH_ACCELERATED_FAMILIES = ("bump",)


class HostModes(NamedTuple):
    """NumPy ``I,V,P`` channel modes for the provisional host route.

    LABEL: ``eq: smooth channel kernel`` [extension]. Shapes are ``(n_ch,)``
    for real ``I,V`` and ``(n_weights,n_ch)`` for complex ``P``. Units follow
    the supplied channel normalisation. Not certified: the numerical error;
    see :class:`HighOrderChannelResult`.
    """

    I: np.ndarray
    V: np.ndarray
    P: np.ndarray


@dataclass(frozen=True)
class HighOrderChannelResult:
    """One fixed-point channel calculation with explicit numerical status.

    LABEL: ``eq: smooth channel kernel`` [extension].

    ``modes`` has ``I,V`` shape ``(n_ch,)`` and complex ``P`` shape
    ``(n_weights,n_ch)``. ``quadrature`` is a fine-minus-coarse *estimate*
    with value shape ``(2+n_weights,n_ch)`` in row order ``I,V,P[:]``; it is
    ``not_applicable`` if all channels use direct summation. ``harmonic_sum``
    is ``unbounded`` whenever an accelerated interval is used; ``bessel`` and
    ``floating_point`` are likewise ``unbounded`` for nonzero emission.
    ``quadrature_converged`` says only whether the displayed difference met
    the requested stopping tolerance, not whether the result is accurate.
    The excluded population tail and physical model discrepancy are outside
    this fixed-point numerical result. The overall numerical result is not
    certified; see the module docstring.
    """

    modes: HostModes
    active_ranges: tuple[tuple[int, int], ...]
    routes: tuple[str, ...]
    evaluated_orders: int
    quadrature_converged: tuple[bool, ...]
    quadrature: ErrorTerm
    harmonic_sum: ErrorTerm
    bessel: ErrorTerm
    floating_point: ErrorTerm


@dataclass(frozen=True)
class HighOrderDerivativeResult:
    """Unscaled physical gamma/B first derivatives of fixed-point channel modes.

    ``d_gamma`` has units of mode power per unit gamma; ``d_B`` has units of
    mode power per Gauss. ``quadrature.value`` (when present) has shape
    ``(3, 2+n_weights, n_ch)`` for value, gamma and B differences. Every
    dense-route difference is a stopping estimate, not an error bound.
    The Bessel and floating-point errors remain unbounded on nonzero lines.
    """

    modes: HostModes
    d_gamma: HostModes
    d_B: HostModes
    active_ranges: tuple[tuple[int, int], ...]
    routes: tuple[str, ...]
    evaluated_orders: int
    quadrature_converged: tuple[bool, ...]
    quadrature: ErrorTerm
    harmonic_sum: ErrorTerm
    bessel: ErrorTerm
    floating_point: ErrorTerm


@dataclass(frozen=True)
class _PreparedHostResponses:
    """Host response closures tied to one immutable channel object."""

    source: Channels
    derivatives: bool
    functions: tuple
    centres: tuple[float, ...]
    supports: tuple[tuple[float, float], ...]
    smooth_edges: tuple[bool, ...]


def _scipy_special():
    try:
        from scipy import special  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "high-order harmonic evaluation requires SciPy; install "
            "syncmoments[high_harmonic]"
        ) from exc
    return special


def _point(gamma, B, mu, eta):
    raw = (gamma, B, mu, eta)
    if any(np.iscomplexobj(v) for v in raw):
        raise ValueError("gamma, B, mu, eta must be real")
    if any(np.asarray(v).ndim != 0 for v in raw):
        raise ValueError("gamma, B, mu, eta must be scalars")
    values = np.asarray(raw, dtype=float)
    if (
        not np.all(np.isfinite(values))
        or gamma < 1.0
        or B < 0.0
        or abs(mu) > 1.0
        or abs(eta) > 1.0
    ):
        raise ValueError("requires finite gamma >= 1, B >= 0, |mu|,|eta| <= 1")
    return tuple(float(v) for v in values)


def _geometry(gamma, mu, eta):
    inv_gamma = 1.0 / gamma
    inv_gamma_sq = inv_gamma**2
    beta = math.sqrt(1.0 - inv_gamma_sq)
    one_minus_beta = inv_gamma_sq / (1.0 + beta)
    b_perp = beta * math.sqrt((1.0 - mu) * (1.0 + mu))
    sin_theta = math.sqrt((1.0 - eta) * (1.0 + eta))
    # 1-beta is evaluated without cancelling two nearly equal numbers.
    if mu >= 0.0 and eta >= 0.0:
        one_minus_mu_eta = (1.0 - mu) + mu * (1.0 - eta)
    elif mu <= 0.0 and eta <= 0.0:
        one_minus_mu_eta = (1.0 + mu) - mu * (1.0 + eta)
    else:
        one_minus_mu_eta = 1.0 - mu * eta
    D = one_minus_beta + beta * one_minus_mu_eta
    if not math.isfinite(D) or D <= 0.0:
        raise ValueError("geometry Doppler denominator is not representable")
    eta_minus_bpar = (eta - mu) + mu * one_minus_beta
    # For the allowed geometry z=x/m <= 1. The positive defect is evaluated
    # through a sum of squares near the Bessel turning region.
    defect = (eta_minus_bpar / D) ** 2 + (inv_gamma * sin_theta / D) ** 2
    direct_z = b_perp * sin_theta / D
    # sqrt(1-defect) preserves 1-z near the turning region but loses z when
    # defect is nearly one; the direct expression is accurate for small z.
    z = direct_z if direct_z < 0.5 else math.sqrt(max(0.0, 1.0 - min(1.0, defect)))
    return beta, b_perp, D, z, eta_minus_bpar


def high_order_line_powers(m, gamma, B, mu, eta):
    """SciPy-evaluated ``(I_m,Q_m,V_m,nu_m)`` at positive real order ``m``.

    Integer ``m`` gives the helical-orbit harmonic line. Real ``m`` is used
    only to interpolate the summand for the accelerated integral. Frequencies
    are Hz, ``B`` is Gauss, and powers are erg/s/sr per electron. The Stokes-V
    sign matches :func:`syncmoments.model._harmonic_cells.harmonic_lines`.
    Orders above ``2**53-2`` are refused: float64 cannot retain their unit
    integer spacing or the neighbouring orders used by the Bessel recurrence.
    Not certified: the Bessel or floating-point error of this call.
    """
    gamma, B, mu, eta = _point(gamma, B, mu, eta)
    if np.iscomplexobj(m):
        raise ValueError("m must be real")
    orders = np.asarray(m, dtype=float)
    if (
        np.any(~np.isfinite(orders))
        or np.any(orders < 1.0)
        or np.any(orders > _MAX_EXACT_FLOAT_INT)
    ):
        raise ValueError(
            "m must contain finite real orders in [1, 2**53-2] with exact unit spacing"
        )
    _, b_perp, D, z, eta_minus_bpar = _geometry(gamma, mu, eta)
    with np.errstate(over="ignore", invalid="ignore"):
        nu_B = E_ESU * B / (2.0 * math.pi * gamma * M_E * C_CGS)
        nu = orders * nu_B / D
    if not np.all(np.isfinite(nu)):
        raise ArithmeticError("nonfinite harmonic frequency")
    if B == 0.0 or b_perp == 0.0:
        zero = np.zeros_like(orders, dtype=float)
        return zero, zero, zero, nu
    special = _scipy_special()
    x = orders * z
    lower = special.jv(orders - 1.0, x)
    upper = special.jv(orders + 1.0, x)
    # J'_m(x) = (J_{m-1}(x) - J_{m+1}(x))/2. Reuse the neighbours already
    # needed for the parallel amplitude rather than evaluating them twice.
    prime = (lower - upper) / 2.0
    a_par = eta_minus_bpar * b_perp * (lower + upper) / (2.0 * D)
    a_perp = b_perp * prime
    omega_B = E_ESU * B / (gamma * M_E * C_CGS)
    if D**3 == 0.0:
        raise ArithmeticError("harmonic prefactor is not representable")
    pref = E_ESU**2 * omega_B**2 / (2.0 * math.pi * C_CGS) * orders**2 / D**3
    I = pref * (a_par**2 + a_perp**2)
    Q = pref * (a_par**2 - a_perp**2)
    V = 2.0 * pref * a_par * a_perp
    if not all(np.all(np.isfinite(v)) for v in (I, Q, V, nu)):
        raise ArithmeticError("nonfinite large-order Bessel or harmonic value")
    return I, Q, V, nu


def _line_powers_with_derivatives(m, gamma, B, mu, eta):
    """Line ``(I,Q,V,nu)`` and unscaled gamma/B derivatives at fixed order.

    The usual branch calls ``jv`` only for orders ``m-1`` and ``m+1``.
    Differentiating their sum/difference by the Bessel equation avoids new
    special-function calls. Very small positive ``z`` uses ``jvp`` because
    those quotient recurrences lose significant digits there; at ``z=0``
    their product with ``z_gamma`` vanishes and is set directly to zero.
    """
    gamma, B, mu, eta = _point(gamma, B, mu, eta)
    if gamma <= 1.0 or B <= 0.0:
        raise ValueError("line derivatives require gamma > 1 and B > 0")
    if np.iscomplexobj(m):
        raise ValueError("m must be real")
    orders = np.asarray(m, dtype=float)
    if (
        np.any(~np.isfinite(orders))
        or np.any(orders < 1.0)
        or np.any(orders > _MAX_EXACT_FLOAT_INT)
    ):
        raise ValueError(
            "m must contain finite real orders in [1, 2**53-2] with exact unit spacing"
        )
    beta, b_perp, D, z, delta = _geometry(gamma, mu, eta)
    nu_B = E_ESU * B / (2.0 * math.pi * gamma * M_E * C_CGS)
    nu = orders * nu_B / D
    if not np.all(np.isfinite(nu)):
        raise ArithmeticError("nonfinite harmonic frequency")
    beta_gamma = gamma**-3 / beta
    D_gamma = -beta_gamma * mu * eta
    nu_gamma = nu * (-1.0 / gamma - D_gamma / D)
    nu_B_derivative = nu / B
    zero = np.zeros_like(orders, dtype=float)
    if b_perp == 0.0:
        return (
            (zero, zero, zero, nu),
            (zero, zero, zero, nu_gamma),
            (
                zero,
                zero,
                zero,
                nu_B_derivative,
            ),
        )

    sin_theta = math.sqrt((1.0 - eta) * (1.0 + eta))
    scaled_delta = delta / D
    scaled_sin = sin_theta / (gamma * D)
    defect = scaled_delta**2 + scaled_sin**2  # stable 1-z² near z=1
    b_perp_gamma = b_perp * beta_gamma / beta
    delta_gamma = -beta_gamma * mu
    if z >= 0.5:
        scaled_delta_gamma = delta_gamma / D - scaled_delta * D_gamma / D
        scaled_sin_gamma = scaled_sin * (-1.0 / gamma - D_gamma / D)
        defect_gamma = 2.0 * (
            scaled_delta * scaled_delta_gamma + scaled_sin * scaled_sin_gamma
        )
        z_gamma = -defect_gamma / (2.0 * z)
    else:
        z_gamma = z * (beta_gamma / beta - D_gamma / D)

    special = _scipy_special()
    x = orders * z
    lower = special.jv(orders - 1.0, x)
    upper = special.jv(orders + 1.0, x)
    S, T = lower + upper, lower - upper
    if z == 0.0:
        S_z = T_z = zero
    elif z < 1e-4:
        lower_z = orders * special.jvp(orders - 1.0, x, 1)
        upper_z = orders * special.jvp(orders + 1.0, x, 1)
        S_z, T_z = lower_z + upper_z, lower_z - upper_z
    else:
        S_z = (orders * T - S) / z
        T_z = (orders * defect * S - T) / z

    parallel_scale = delta * b_perp / (2.0 * D)
    parallel_scale_gamma = (delta_gamma * b_perp + delta * b_perp_gamma) / (
        2.0 * D
    ) - parallel_scale * D_gamma / D
    a_par = parallel_scale * S
    a_perp = b_perp * T / 2.0
    a_par_gamma = parallel_scale_gamma * S + parallel_scale * S_z * z_gamma
    a_perp_gamma = (b_perp_gamma * T + b_perp * T_z * z_gamma) / 2.0
    omega_B = E_ESU * B / (gamma * M_E * C_CGS)
    if D**3 == 0.0:
        raise ArithmeticError("harmonic prefactor is not representable")
    pref = E_ESU**2 * omega_B**2 / (2.0 * math.pi * C_CGS) * orders**2 / D**3
    pref_gamma = pref * (-2.0 / gamma - 3.0 * D_gamma / D)
    par_sq, perp_sq = a_par**2, a_perp**2
    I = pref * (par_sq + perp_sq)
    Q = pref * (par_sq - perp_sq)
    V = 2.0 * pref * a_par * a_perp
    par_cross, perp_cross = 2.0 * a_par * a_par_gamma, 2.0 * a_perp * a_perp_gamma
    I_gamma = pref_gamma * (par_sq + perp_sq) + pref * (par_cross + perp_cross)
    Q_gamma = pref_gamma * (par_sq - perp_sq) + pref * (par_cross - perp_cross)
    V_gamma = 2.0 * (
        pref_gamma * a_par * a_perp
        + pref * (a_par_gamma * a_perp + a_par * a_perp_gamma)
    )
    I_B, Q_B, V_B = 2.0 * I / B, 2.0 * Q / B, 2.0 * V / B
    result = (
        (I, Q, V, nu),
        (I_gamma, Q_gamma, V_gamma, nu_gamma),
        (
            I_B,
            Q_B,
            V_B,
            nu_B_derivative,
        ),
    )
    if not all(np.all(np.isfinite(part)) for row in result for part in row):
        raise ArithmeticError("nonfinite large-order line or derivative value")
    return result


@lru_cache(maxsize=12)
def _leggauss(order):
    # SciPy avoids NumPy leggauss's dense eigenproblem for large rules.
    return _scipy_special().roots_legendre(order)


def _single_channel(channels, j):
    """One response without evaluating all other channels at every order."""
    table = None
    if channels.table is not None:
        grid, rows = channels.table
        table = (grid, (rows[j],))
    return dataclasses.replace(
        channels,
        centres_hz=channels.centres_hz[j : j + 1],
        widths_hz=channels.widths_hz[j : j + 1],
        support=channels.support[j : j + 1],
        nodes=channels.nodes[j : j + 1],
        weights=channels.weights[j : j + 1],
        table=table,
    )


def _host_response(channels, j):
    """One host response; use NumPy for the common analytic bump family."""
    if channels.family != "bump":
        return _single_channel(channels, j)
    centre = float(channels.centres_hz[j])
    width = float(channels.widths_hz[j])
    lo, hi = (float(value) for value in np.asarray(channels.support[j]))
    scale = (
        1.0
        if channels.normalisation == "unit_peak"
        else 1.0
        / (width * unit_peak_area("bump", channels.taper, channels.support_sigma))
    )

    def response(frequency):
        frequency = np.asarray(frequency, dtype=float)
        inside = (frequency > lo) & (frequency < hi)
        t = np.where(inside, (frequency - centre) / width, 0.0)
        shape = np.exp(1.0 - 1.0 / np.maximum(1.0 - t * t, FLOOR))
        return np.where(inside, shape, 0.0) * scale

    return response


def _host_bump_response_and_slope_function(channels, j):
    """Prepare one bump and frequency slope for repeated quadrature calls."""
    response_function = _host_response(channels, j)
    centre = float(channels.centres_hz[j])
    width = float(channels.widths_hz[j])
    lo, hi = (float(value) for value in np.asarray(channels.support[j]))

    def response_and_slope(frequency):
        frequency = np.asarray(frequency, dtype=float)
        response = response_function(frequency)
        inside = (frequency > lo) & (frequency < hi)
        t = np.where(inside, (frequency - centre) / width, 0.0)
        q = (1.0 - t) * (1.0 + t)
        active = inside & (q > FLOOR) & (response > 0.0)
        safe_q = np.where(active, q, 1.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            slope = np.where(active, response * (-2.0 * t / (width * safe_q**2)), 0.0)
        return response, slope

    return response_and_slope


def _host_bump_response_and_slope(channels, j, frequency):
    """Stored bump and dR/dnu, including the flat zero at each open edge."""
    return _host_bump_response_and_slope_function(channels, j)(frequency)


def _prepare_host_responses(channels, *, derivatives):
    """Extract channel scalars once for a repeated angular preparation loop."""
    factory = _host_bump_response_and_slope_function if derivatives else _host_response
    centres = tuple(float(v) for v in np.asarray(channels.centres_hz))
    widths = tuple(float(v) for v in np.asarray(channels.widths_hz))
    supports = tuple(
        tuple(float(v) for v in row) for row in np.asarray(channels.support)
    )
    smooth_edges = (
        tuple(
            Fraction.from_float(lo)
            <= Fraction.from_float(centre) - Fraction.from_float(width)
            and Fraction.from_float(hi)
            >= Fraction.from_float(centre) + Fraction.from_float(width)
            for centre, width, (lo, hi) in zip(centres, widths, supports, strict=True)
        )
        if channels.family == "bump"
        else tuple(False for _ in centres)
    )
    return _PreparedHostResponses(
        channels,
        derivatives,
        tuple(factory(channels, j) for j in range(channels.n_ch)),
        centres,
        supports,
        smooth_edges,
    )


def _response_functions(channels, prepared, *, derivatives):
    if prepared is None:
        prepared = _prepare_host_responses(channels, derivatives=derivatives)
    if (
        not isinstance(prepared, _PreparedHostResponses)
        or prepared.source is not channels
        or prepared.derivatives is not derivatives
        or len(prepared.functions) != channels.n_ch
        or len(prepared.centres) != channels.n_ch
        or len(prepared.supports) != channels.n_ch
        or len(prepared.smooth_edges) != channels.n_ch
    ):
        raise ValueError("prepared host responses do not match the channels")
    if derivatives and not all(prepared.smooth_edges):
        raise ValueError("bump support must contain its exact smooth zeros")
    return prepared


def _host_phase_weights(phase, frequency, depth_ref, s_depth):
    """Avoid host-to-JAX dispatch for the default and Taylor phase routes."""
    if phase is None:
        return np.ones((1, frequency.size), dtype=complex)
    if isinstance(phase, TaylorPhase):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            tau = 2.0 * (C_SI_M / frequency) ** 2
            base = np.exp(1j * tau * depth_ref)
            step = 1j * tau * s_depth
            terms = [np.ones_like(base)]
            for degree in range(1, phase.n_weights):
                terms.append(terms[-1] * step / degree)
            return base[None, :] * np.stack(terms)
    return np.asarray(
        phase_weights(phase, phase_coordinate(frequency), depth_ref, s_depth)
    )


def _host_taylor_phase_and_slope(phase, frequency, depth_ref, s_depth):
    """Taylor weights and their frequency derivative, including b=0."""
    weights = _host_phase_weights(phase, frequency, depth_ref, s_depth)
    if phase is None:
        return weights, np.zeros_like(weights)
    tau = 2.0 * (C_SI_M / frequency) ** 2
    degree = np.arange(phase.n_weights, dtype=float)[:, None]
    slope = (
        weights * (-2.0 / frequency)[None, :] * (1j * tau[None, :] * depth_ref + degree)
    )
    return weights, slope


def _active_range(support, spacing):
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("harmonic frequency spacing is not finite positive")
    lo, hi = (float(v) for v in support)
    if not (math.isfinite(lo / spacing) and math.isfinite(hi / spacing)):
        raise ValueError("channel support order is not representable")
    # The response is zero outside its open support. Pad both endpoints by
    # two orders because floating division can round an exact edge across an
    # integer; the extra terms are evaluated with the actual response.
    a = max(1, math.floor(lo / spacing) - 2)
    b = math.ceil(hi / spacing) + 2
    if b > _MAX_EXACT_FLOAT_INT:
        raise ValueError("active order exceeds exact unit spacing of float64 (2^53)")
    return a, b


def _validate_settings(direct_limit, quad_order, max_quad_order, rtol, atol):
    for name, value, minimum in (
        ("direct_limit", direct_limit, 1),
        ("quad_order", quad_order, 4),
        ("max_quad_order", max_quad_order, 8),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if direct_limit > 1_000_000:
        raise ValueError("direct_limit exceeds the one-million-line resource guard")
    if max_quad_order < 2 * quad_order:
        raise ValueError("max_quad_order must be at least 2*quad_order")
    if max_quad_order > 8192:
        raise ValueError("max_quad_order exceeds the 8192-node resource guard")
    if not (math.isfinite(rtol) and math.isfinite(atol) and rtol >= 0 and atol >= 0):
        raise ValueError("rtol and atol must be finite nonnegative numbers")


def high_order_channel_modes(
    channels: Channels,
    gamma,
    B,
    mu,
    eta,
    *,
    phase=None,
    depth_ref=0.0,
    s_depth=1.0,
    direct_limit=4096,
    quad_order=32,
    max_quad_order=256,
    rtol=1e-6,
    atol=0.0,
    allow_unconverged=False,
    _prepared_responses=None,
) -> HighOrderChannelResult:
    """Evaluate compact-channel ``I,V,P`` at one ``(gamma,B,mu,eta)`` point.

    The whole positive-order support is included; no ``m_max`` tail is cut.
    The direct/integral choice uses the number of candidate integers strictly
    inside each channel's support. Direct evaluation also computes up to six
    endpoint-padding orders whose response should vanish. Above
    ``direct_limit`` candidate integers, only ``bump`` responses use the corrected
    integral. The method uses DLMF 2.10.1 (link in module docstring), with
    its remainder left ``unbounded``. ``rtol``/``atol`` stop refinement of a
    quadrature *estimate* only; they do not certify the harmonic sum. Direct
    sums retain unbounded special-function and floating-point error.
    Failure to meet that estimate raises ``ArithmeticError`` by default;
    ``allow_unconverged=True`` returns the value with a false convergence flag
    solely for exploratory diagnostics. Broad order ranges with ``b/a > 16``
    are refused because a whole-interval rule can miss localised emission.

    This is a host-only, value-only API. It is not a drop-in ``HarmonicKernel``
    replacement and cannot feed certified ``build_basis`` error budgets.
    """
    if not isinstance(channels, Channels):
        raise TypeError("channels must be a Channels instance")
    gamma, B, mu, eta = _point(gamma, B, mu, eta)
    _validate_settings(direct_limit, quad_order, max_quad_order, rtol, atol)
    if not isinstance(allow_unconverged, bool):
        raise ValueError("allow_unconverged must be a bool")
    if not (math.isfinite(depth_ref) and math.isfinite(s_depth) and s_depth > 0):
        raise ValueError("depth_ref must be finite and s_depth finite positive")
    n_ch = channels.n_ch
    n_weights = 1 if phase is None else phase.n_weights
    values = np.zeros((2 + n_weights, n_ch), dtype=complex)
    qdiff = np.zeros_like(values.real)
    if B == 0.0 or gamma == 1.0 or abs(mu) == 1.0:
        return HighOrderChannelResult(
            HostModes(values[0].real, values[1].real, values[2:]),
            tuple((0, -1) for _ in range(n_ch)),
            tuple("zero" for _ in range(n_ch)),
            0,
            tuple(True for _ in range(n_ch)),
            ErrorTerm.not_applicable("zero emissivity"),
            ErrorTerm.not_applicable("zero emissivity"),
            ErrorTerm.not_applicable("zero emissivity"),
            ErrorTerm.not_applicable("zero emissivity"),
        )
    _, _, D, _, _ = _geometry(gamma, mu, eta)
    spacing = E_ESU * B / (2.0 * math.pi * gamma * M_E * C_CGS * D)
    prepared = _response_functions(channels, _prepared_responses, derivatives=False)
    supports = prepared.supports
    active_ranges = tuple(_active_range(s, spacing) for s in supports)
    responses = prepared.functions
    routes = []
    converged = []
    evaluations = 0

    def terms(j, orders):
        nonlocal evaluations
        orders = np.atleast_1d(np.asarray(orders, dtype=float))
        evaluations += orders.size
        I, Q, V, nu = high_order_line_powers(orders, gamma, B, mu, eta)
        response = np.asarray(responses[j](nu))
        if response.ndim > 1:
            response = response[0]
        weights = _host_phase_weights(phase, nu, depth_ref, s_depth)
        weighted = np.concatenate(
            (
                (response * I)[None, :],
                (response * V)[None, :],
                weights * (response * Q)[None, :],
            ),
            axis=0,
        )
        if not np.all(np.isfinite(weighted)):
            raise ArithmeticError("nonfinite channel term from response or phase")
        return weighted

    def integral(j, a, b, order, endpoint):
        nodes, weights = _leggauss(order)
        half = (b - a) / 2.0
        middle = a + half
        sample = middle + half * nodes
        area = np.sum(terms(j, sample) * (half * weights)[None, :], axis=1)
        # DLMF 2.10.1: integral plus half of each endpoint. The derivative
        # remainder is deliberately not discarded from the error status.
        return area + endpoint

    for j, (a, b) in enumerate(active_ranges):
        lo, hi = map(float, supports[j])
        candidate_count = max(0, math.ceil(hi / spacing) - math.floor(lo / spacing) - 1)
        if candidate_count <= direct_limit:
            values[:, j] = np.sum(terms(j, np.arange(a, b + 1, dtype=float)), axis=1)
            routes.append("direct")
            converged.append(True)
            continue
        if channels.family not in _SMOOTH_ACCELERATED_FAMILIES:
            raise ValueError(
                "accelerated range requires a bump response; "
                "increase direct_limit for this channel"
            )
        if not prepared.smooth_edges[j]:
            raise ValueError(
                "accelerated bump support must contain its exact smooth zeros"
            )
        if b > 16 * a:
            raise ValueError(
                "accelerated range spans too many order scales for one integral; "
                "use narrower channels or a future segmented evaluator"
            )
        endpoint = 0.5 * (terms(j, [a])[:, 0] + terms(j, [b])[:, 0])
        coarse = integral(j, a, b, quad_order, endpoint)
        order = quad_order * 2
        while True:
            fine = integral(j, a, b, order, endpoint)
            difference = np.abs(fine - coarse)
            good = bool(np.all(difference <= atol + rtol * np.abs(fine)))
            if good or order * 2 > max_quad_order:
                if not good and not allow_unconverged:
                    raise ArithmeticError(
                        f"channel {j} quadrature difference did not meet the "
                        "requested tolerance; refine the rule or set "
                        "allow_unconverged=True for an exploratory result"
                    )
                values[:, j] = fine
                qdiff[:, j] = difference
                routes.append("corrected_integral")
                converged.append(good)
                break
            coarse, order = fine, order * 2

    any_accel = "corrected_integral" in routes
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(qdiff)):
        raise ArithmeticError("nonfinite channel sum or quadrature difference")
    return HighOrderChannelResult(
        HostModes(values[0].real, values[1].real, values[2:]),
        active_ranges,
        tuple(routes),
        evaluations,
        tuple(converged),
        (
            ErrorTerm(
                qdiff,
                "estimate",
                "absolute difference of the last two Gauss-Legendre orders; "
                "zero columns used direct summation",
                "E_num",
            )
            if any_accel
            else ErrorTerm.not_applicable("direct summation only")
        ),
        (
            ErrorTerm.unbounded(
                "Euler--Maclaurin derivative remainder from DLMF 2.10.1 not bounded",
                "E_num",
            )
            if any_accel
            else ErrorTerm.not_applicable("direct summation only")
        ),
        ErrorTerm.unbounded("SciPy jv error not bounded", "E_num"),
        ErrorTerm.unbounded(
            "float64 support, phase, and sum roundoff not bounded", "E_num"
        ),
    )


def high_order_channel_derivatives(
    channels: Channels,
    gamma,
    B,
    mu,
    eta,
    *,
    phase=None,
    depth_ref=0.0,
    s_depth=1.0,
    direct_limit=4096,
    quad_order=32,
    max_quad_order=256,
    rtol=1e-6,
    atol=0.0,
    atol_gamma=0.0,
    atol_B=0.0,
    allow_unconverged=False,
    _prepared_responses=None,
) -> HighOrderDerivativeResult:
    """Value and unscaled physical gamma/B derivatives at one fixed point.

    Requires a stored ``bump`` response, positive B, gamma above one, and
    ``TaylorPhase`` or no phase. Direct versus dense selection still uses the
    channel's active integer count. The dense stopping check covers *all*
    value and derivative rows. ``atol``, ``atol_gamma`` and ``atol_B`` are
    separate absolute stopping thresholds in value, per-gamma and per-Gauss
    units respectively; all default to zero. No threshold is an error bound.
    The Euler--Maclaurin, SciPy and floating-point errors remain unbounded.
    """
    if not isinstance(channels, Channels):
        raise TypeError("channels must be a Channels instance")
    gamma, B, mu, eta = _point(gamma, B, mu, eta)
    if gamma <= 1.0 or B <= 0.0:
        raise ValueError("channel derivatives require gamma > 1 and B > 0")
    if channels.family != "bump":
        raise ValueError("channel derivatives require a bump response")
    if phase is not None and not isinstance(phase, TaylorPhase):
        raise ValueError("channel derivatives require TaylorPhase or no phase")
    _validate_settings(direct_limit, quad_order, max_quad_order, rtol, atol)
    if not (
        math.isfinite(atol_gamma)
        and math.isfinite(atol_B)
        and atol_gamma >= 0
        and atol_B >= 0
    ):
        raise ValueError("atol_gamma and atol_B must be finite nonnegative numbers")
    if not isinstance(allow_unconverged, bool):
        raise ValueError("allow_unconverged must be a bool")
    if not (math.isfinite(depth_ref) and math.isfinite(s_depth) and s_depth > 0):
        raise ValueError("depth_ref must be finite and s_depth finite positive")
    n_ch = channels.n_ch
    n_weights = 1 if phase is None else phase.n_weights
    values = np.zeros((3, 2 + n_weights, n_ch), dtype=complex)
    qdiff = np.zeros_like(values.real)
    absolute_limits = np.asarray((atol, atol_gamma, atol_B))[:, None]

    def modes(rows):
        return HostModes(rows[0].real, rows[1].real, rows[2:])

    if abs(mu) == 1.0:
        return HighOrderDerivativeResult(
            modes(values[0]),
            modes(values[1]),
            modes(values[2]),
            tuple((0, -1) for _ in range(n_ch)),
            tuple("zero" for _ in range(n_ch)),
            0,
            tuple(True for _ in range(n_ch)),
            ErrorTerm.not_applicable("zero emissivity on pitch axis"),
            ErrorTerm.not_applicable("zero emissivity on pitch axis"),
            ErrorTerm.not_applicable("zero emissivity on pitch axis"),
            ErrorTerm.not_applicable("zero emissivity on pitch axis"),
        )
    _, _, D, _, _ = _geometry(gamma, mu, eta)
    spacing = E_ESU * B / (2.0 * math.pi * gamma * M_E * C_CGS * D)
    prepared = _response_functions(channels, _prepared_responses, derivatives=True)
    supports = prepared.supports
    active_ranges = tuple(_active_range(s, spacing) for s in supports)
    responses = prepared.functions
    routes = []
    converged = []
    evaluations = 0

    def terms(j, orders):
        nonlocal evaluations
        orders = np.atleast_1d(np.asarray(orders, dtype=float))
        evaluations += orders.size
        line, gamma_line, B_line = _line_powers_with_derivatives(
            orders, gamma, B, mu, eta
        )
        I, Q, V, frequency = line
        I_gamma, Q_gamma, V_gamma, frequency_gamma = gamma_line
        I_B, Q_B, V_B, frequency_B = B_line
        response, response_nu = responses[j](frequency)
        # Outside the open support both response and derivative are zero. Use
        # a benign phase frequency there to avoid 0*inf in padded orders.
        phase_frequency = np.where(
            (response != 0.0) | (response_nu != 0.0),
            frequency,
            prepared.centres[j],
        )
        weights, weights_nu = _host_taylor_phase_and_slope(
            phase, phase_frequency, depth_ref, s_depth
        )
        response_gamma = response_nu * frequency_gamma
        response_B = response_nu * frequency_B
        weighted = np.empty((3, 2 + n_weights, orders.size), dtype=complex)
        weighted[0, 0] = response * I
        weighted[0, 1] = response * V
        weighted[0, 2:] = weights * (response * Q)[None, :]
        weighted[1, 0] = response * I_gamma + response_gamma * I
        weighted[1, 1] = response * V_gamma + response_gamma * V
        weighted[1, 2:] = (
            weights * (response * Q_gamma + response_gamma * Q)[None, :]
            + weights_nu * (frequency_gamma * response * Q)[None, :]
        )
        weighted[2, 0] = response * I_B + response_B * I
        weighted[2, 1] = response * V_B + response_B * V
        weighted[2, 2:] = (
            weights * (response * Q_B + response_B * Q)[None, :]
            + weights_nu * (frequency_B * response * Q)[None, :]
        )
        if not np.all(np.isfinite(weighted)):
            raise ArithmeticError("nonfinite channel derivative term")
        return weighted

    def integral(j, a, b, order, endpoint):
        nodes, weights = _leggauss(order)
        half = (b - a) / 2.0
        middle = a + half
        sample = middle + half * nodes
        area = np.sum(terms(j, sample) * (half * weights)[None, None, :], axis=2)
        return area + endpoint

    for j, (a, b) in enumerate(active_ranges):
        lo, hi = map(float, supports[j])
        candidate_count = max(0, math.ceil(hi / spacing) - math.floor(lo / spacing) - 1)
        if candidate_count <= direct_limit:
            values[:, :, j] = np.sum(terms(j, np.arange(a, b + 1, dtype=float)), axis=2)
            routes.append("direct")
            converged.append(True)
            continue
        if b > 16 * a:
            raise ValueError(
                "accelerated range spans too many order scales for one integral; "
                "use narrower channels or a future segmented evaluator"
            )
        endpoint = 0.5 * (terms(j, [a])[:, :, 0] + terms(j, [b])[:, :, 0])
        coarse = integral(j, a, b, quad_order, endpoint)
        order = quad_order * 2
        while True:
            fine = integral(j, a, b, order, endpoint)
            difference = np.abs(fine - coarse)
            good = bool(np.all(difference <= absolute_limits + rtol * np.abs(fine)))
            if good or order * 2 > max_quad_order:
                if not good and not allow_unconverged:
                    raise ArithmeticError(
                        f"channel {j} derivative quadrature difference did not meet "
                        "the requested tolerance; refine the rule or set "
                        "allow_unconverged=True for an exploratory result"
                    )
                values[:, :, j] = fine
                qdiff[:, :, j] = difference
                routes.append("corrected_integral")
                converged.append(good)
                break
            coarse, order = fine, order * 2

    any_accel = "corrected_integral" in routes
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(qdiff)):
        raise ArithmeticError("nonfinite channel derivative or quadrature difference")
    return HighOrderDerivativeResult(
        modes(values[0]),
        modes(values[1]),
        modes(values[2]),
        active_ranges,
        tuple(routes),
        evaluations,
        tuple(converged),
        (
            ErrorTerm(
                qdiff,
                "estimate",
                "absolute differences of last two Gauss-Legendre rules for "
                "value/gamma/B rows; not bounds",
                "E_num",
            )
            if any_accel
            else ErrorTerm.not_applicable("direct summation only")
        ),
        (
            ErrorTerm.unbounded(
                "Euler--Maclaurin derivative remainder from DLMF 2.10.1 not bounded",
                "E_num",
            )
            if any_accel
            else ErrorTerm.not_applicable("direct summation only")
        ),
        ErrorTerm.unbounded("SciPy jv/jvp error not bounded", "E_num"),
        ErrorTerm.unbounded(
            "float64 support, derivative, phase, and sum roundoff not bounded",
            "E_num",
        ),
    )


__all__ = [
    "HostModes",
    "HighOrderChannelResult",
    "HighOrderDerivativeResult",
    "high_order_line_powers",
    "high_order_channel_modes",
    "high_order_channel_derivatives",
]
