"""Conditional interval enclosure for sparse fixed-point harmonic channels.

This internal route encloses the finite-gamma integer-line reference for a
single exact binary-float particle point and unit-peak bump responses. It
uses interval geometry, an analytic integer-Bessel contour remainder, and
interval response/Faraday evaluation. A dense channel can return zero with a
bound from DLMF 10.14.5--6 when the whole channel is below a requested
absolute tolerance. Angular and parameter derivatives are rejected. The
enclosure relies on mpmath's
experimental interval elementary functions, so it is not yet a production
certificate for a population or a build_basis result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..constants import C_CGS, C_SI_M, E_ESU, M_E
from ._verified_airy import IntegerAiryEnclosure, integer_bessel_airy_enclosure
from ._verified_bessel import IntegerBesselEnclosure, integer_bessel_enclosure
from .channels import Channels
from .errors import ErrorTerm
from .high_order_harmonic import HostModes, _point
from .phase import TaylorPhase


@dataclass(frozen=True)
class SparseChannelEnclosure:
    """Fixed-point modes and componentwise absolute interval errors."""

    modes: HostModes
    active_ranges: tuple[tuple[int, int], ...]
    routes: tuple[str, ...]
    evaluated_orders: int
    bessel_methods: tuple[tuple[int, str], ...]
    absolute_error: ErrorTerm
    bessel: ErrorTerm
    interval_arithmetic: ErrorTerm
    harmonic_sum: ErrorTerm


def _outward_float(value) -> float:
    if value.b == 0:
        return 0.0
    result = float(value.b)
    if not math.isfinite(result):
        raise ArithmeticError("interval error cannot be represented as a finite float")
    return math.nextafter(max(0.0, result), math.inf)


def _real_error(ctx, full, approximate):
    centre = float(approximate.mid)
    if not math.isfinite(centre):
        raise ArithmeticError("interval midpoint is not a finite float")
    centred = ctx.mpf(centre)
    total = max((centred - full.a).b, (full.b - centred).b)
    bessel = max((full.b - approximate.a).b, (approximate.b - full.a).b)
    rounding = max((centred - approximate.a).b, (approximate.b - centred).b)
    return (
        centre,
        _outward_float(total),
        _outward_float(bessel),
        _outward_float(rounding),
    )


def _component_error(ctx, full, approximate, *, complex_value=False):
    if not complex_value:
        return _real_error(ctx, full, approximate)
    real = _real_error(ctx, full.real, approximate.real)
    imag = _real_error(ctx, full.imag, approximate.imag)
    # |x+iy| <= |x|+|y|, rounded outward for the published float bound.
    errors = tuple(math.nextafter(real[k] + imag[k], math.inf) for k in (1, 2, 3))
    return complex(real[0], imag[0]), *errors


def _response(ctx, channels, j, frequency):
    """Enclose the ideal unit-peak bump over a frequency interval.

    The stored binary support, centre and width define the reference shape.
    For ``q=((nu-centre)/width)**2 < 1``, ``exp(1-1/(1-q))`` decreases
    monotonically in ``q``. This real-valued profile differs from the host
    floating-point FLOOR safeguard, whose roundoff is not enclosed here.
    """
    lo, hi = (ctx.mpf(float(v)) for v in np.asarray(channels.support[j]))
    if frequency.b <= lo or frequency.a >= hi:
        return ctx.mpf(0)
    centre = ctx.mpf(float(channels.centres_hz[j]))
    width = ctx.mpf(float(channels.widths_hz[j]))
    clipped_lo = max(frequency.a, lo)
    clipped_hi = min(frequency.b, hi)
    q_lo = ((clipped_lo - centre) / width) ** 2
    q_hi = ((clipped_hi - centre) / width) ** 2
    centre_possible = clipped_lo.a <= centre.b and centre.a <= clipped_hi.b
    q_min = ctx.mpf(0) if centre_possible else min(q_lo.a, q_hi.a)
    q_max = max(q_lo.b, q_hi.b)
    if q_min.a >= 1:
        return ctx.mpf(0)
    upper_denominator = 1 - q_min
    upper = (
        ctx.mpf(1)
        if not upper_denominator.a > 0
        else ctx.exp(1 - 1 / upper_denominator).b
    )
    # Endpoints on or outside the open support have exactly zero response.
    lower_denominator = 1 - q_max
    if (
        frequency.a <= lo
        or frequency.b >= hi
        or q_max.b >= 1
        or not lower_denominator.a > 0
    ):
        lower = ctx.mpf(0)
    else:
        lower = ctx.exp(1 - 1 / lower_denominator).a
    lower = max(ctx.mpf(0), lower)
    upper = min(ctx.mpf(1), upper)
    if lower > upper:
        raise ArithmeticError("bump interval bounds are inconsistent")
    return ctx.mpf([lower, upper])


def _geometry(ctx, gamma, B, mu, eta):
    g, field, pitch, view = (ctx.mpf(v) for v in (gamma, B, mu, eta))
    inverse = 1 / g
    beta = ctx.sqrt(1 - inverse**2)
    one_minus_beta = inverse**2 / (1 + beta)
    if mu >= 0 and eta >= 0:
        one_minus_product = (1 - pitch) + pitch * (1 - view)
    elif mu <= 0 and eta <= 0:
        one_minus_product = (1 + pitch) - pitch * (1 + view)
    else:
        one_minus_product = 1 - pitch * view
    D = one_minus_beta + beta * one_minus_product
    delta = (view - pitch) + pitch * one_minus_beta
    perpendicular = beta * ctx.sqrt((1 - pitch) * (1 + pitch))
    viewing_sine = ctx.sqrt((1 - view) * (1 + view))
    z = perpendicular * viewing_sine / D
    omega_b = ctx.mpf(E_ESU) * field / (g * ctx.mpf(M_E) * ctx.mpf(C_CGS))
    spacing = omega_b / (2 * ctx.pi * D)
    return D, delta, perpendicular, z, omega_b, spacing


def _stokes(ctx, order, geometry, j_value, derivative):
    D, delta, perpendicular, z, omega_b, _ = geometry
    # DLMF 10.6.1: J_(m-1)(mz)+J_(m+1)(mz)=2*J_m(mz)/z.
    # https://dlmf.nist.gov/10.6.E1
    a_parallel = delta * perpendicular * j_value / (D * z)
    a_perpendicular = perpendicular * derivative
    prefactor = (
        ctx.mpf(E_ESU) ** 2
        * omega_b**2
        * order**2
        / (2 * ctx.pi * ctx.mpf(C_CGS) * D**3)
    )
    I = prefactor * (a_parallel**2 + a_perpendicular**2)
    Q = prefactor * (a_parallel**2 - a_perpendicular**2)
    V = 2 * prefactor * a_parallel * a_perpendicular
    return I, Q, V


def _faint_channel_bounds(ctx, geometry, first, last, frequency_lo, n_weights, s_depth):
    """Bound an entire order range without evaluating its Bessel lines.

    DLMF 10.14.5--6 bound J_m(mz) and J'_m(mz) for 0<z<=1. DLMF 10.6.1
    converts J_m/z to the neighbouring-order amplitude. Since |Q|,|V|<=I,
    a maximum over the range times its integer count bounds every Stokes
    sum. The unit-peak bump response is between zero and one. This coarse
    envelope is intended only for channels already faint at the requested
    *absolute* scale; it makes no relative-accuracy claim.
    """
    if last < first:
        return [ctx.mpf(0) for _ in range(2 + n_weights)]
    D, delta, perpendicular, z, omega_b, _ = geometry
    z_lo, z_hi = z.a, z.b
    if z_hi == 1:
        xi_lower = ctx.mpf(0)
    else:
        root = ctx.sqrt((1 - z_hi) * (1 + z_hi))
        xi = ctx.ln((1 + root) / z_hi) - root
        xi_lower = max(ctx.mpf(0), xi.a)
    K_upper = (ctx.sqrt(ctx.sqrt(1 + z_hi**2)) / (z_lo * ctx.sqrt(2 * ctx.pi))).b
    delta_upper = max(abs(delta.a), abs(delta.b))
    A_upper = (delta_upper * perpendicular.b / (D.a * z_lo)).b
    C_upper = (
        ctx.mpf(E_ESU) ** 2 * omega_b**2 / (2 * ctx.pi * ctx.mpf(C_CGS) * D**3)
    ).b
    count = last - first + 1
    intensity = (
        C_upper
        * (A_upper**2 * last**2 + perpendicular.b**2 * K_upper**2 * last)
        * count
        * ctx.exp(-2 * first * xi_lower)
    ).b
    tau_max = (2 * (ctx.mpf(C_SI_M) / frequency_lo) ** 2).b
    bounds = [intensity, intensity]
    weight = ctx.mpf(1)
    for degree in range(n_weights):
        if degree:
            weight = (weight * tau_max * abs(s_depth) / degree).b
        bounds.append((intensity * weight).b)
    return bounds


def certified_sparse_channel_modes(
    channels: Channels,
    gamma,
    B,
    mu,
    eta,
    *,
    phase: TaylorPhase | None = None,
    depth_ref=0.0,
    s_depth=1.0,
    max_modes: int = 16,
    bessel_tolerance: float = 1e-10,
    decimal_digits: int = 70,
    max_bessel_nodes: int = 200_000,
    atol: float | None = None,
) -> SparseChannelEnclosure:
    """Enclose sparse unit-peak bump channel modes at one fixed particle point.

    The returned bounds refer to the line model with the exact binary values
    of all supplied floats and stored channel parameters. They exclude
    physical-model error and uncertainty in those inputs. Only a TaylorPhase
    or no phase is supported. The selected integer interval is obtained from
    outward-rounded interval line spacing; all possibly active lines are
    considered, and proven zero-response lines skip Bessel evaluation. When
    ``atol`` is supplied, a channel exceeding ``max_modes`` can instead use
    a whole-range DLMF magnitude bound if every output component meets it.
    """
    gamma, B, mu, eta = _point(gamma, B, mu, eta)
    if not isinstance(channels, Channels) or channels.family != "bump":
        raise ValueError("sparse enclosure requires bump Channels")
    if channels.normalisation != "unit_peak":
        raise ValueError("sparse enclosure requires unit_peak normalisation")
    if phase is not None and not isinstance(phase, TaylorPhase):
        raise ValueError("sparse enclosure supports TaylorPhase or no phase")
    if not math.isfinite(depth_ref) or not math.isfinite(s_depth) or s_depth <= 0:
        raise ValueError("depth_ref must be finite and s_depth finite positive")
    depth_ref, s_depth = float(depth_ref), float(s_depth)
    if isinstance(max_modes, bool) or not isinstance(max_modes, int) or max_modes < 1:
        raise ValueError("max_modes must be an integer >= 1")
    if (
        isinstance(decimal_digits, bool)
        or not isinstance(decimal_digits, int)
        or decimal_digits < 30
    ):
        raise ValueError("decimal_digits must be an integer >= 30")
    if (
        isinstance(bessel_tolerance, bool)
        or not math.isfinite(bessel_tolerance)
        or bessel_tolerance <= 0
    ):
        raise ValueError("bessel_tolerance must be finite positive")
    if atol is not None and (
        isinstance(atol, bool) or not math.isfinite(atol) or atol < 0
    ):
        raise ValueError("atol must be finite nonnegative or None")
    try:
        from mpmath.ctx_iv import MPIntervalContext  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("sparse enclosures require mpmath") from exc

    ctx = MPIntervalContext()
    ctx.dps = decimal_digits
    n_weights = 1 if phase is None else phase.n_weights
    shape = (2 + n_weights, channels.n_ch)
    values = np.zeros(shape, dtype=complex)
    total_error = np.zeros(shape, dtype=float)
    bessel_error = np.zeros(shape, dtype=float)
    arithmetic_error = np.zeros(shape, dtype=float)
    zero = ctx.mpf(0)
    full = [
        [ctx.mpf(0) if row < 2 else ctx.mpc(0) for _ in range(channels.n_ch)]
        for row in range(2 + n_weights)
    ]
    approximate = [
        [ctx.mpf(0) if row < 2 else ctx.mpc(0) for _ in range(channels.n_ch)]
        for row in range(2 + n_weights)
    ]
    if B == 0 or gamma == 1 or abs(mu) == 1:
        return SparseChannelEnclosure(
            HostModes(values[0].real, values[1].real, values[2:]),
            tuple((0, -1) for _ in range(channels.n_ch)),
            tuple("zero" for _ in range(channels.n_ch)),
            0,
            (),
            ErrorTerm(total_error, "bound", "zero emissivity", "E_num"),
            ErrorTerm(bessel_error, "bound", "zero emissivity", "E_num"),
            ErrorTerm(arithmetic_error, "bound", "zero emissivity", "E_num"),
            ErrorTerm(total_error, "bound", "zero emissivity", "E_num"),
        )

    geometry = _geometry(ctx, gamma, B, mu, eta)
    D, _, _, z, omega_b, spacing = geometry
    if not (z.a >= ctx.mpf("0.5") and z.b <= 1):
        raise ValueError("sparse enclosure requires 0.5 <= z <= 1")
    supports = np.asarray(channels.support, dtype=float)
    ranges: list[tuple[int, int]] = []
    routes: list[str] = []
    dense_bounds = np.zeros(shape, dtype=float)
    for lo, hi in supports:
        lower = ctx.mpf(float(lo)) / spacing
        upper = ctx.mpf(float(hi)) / spacing
        lower_float, upper_float = float(lower.a), float(upper.b)
        if not (math.isfinite(lower_float) and math.isfinite(upper_float)):
            raise ValueError("active order is not representable")
        lower_order = math.floor(math.nextafter(lower_float, -math.inf))
        upper_order = math.ceil(math.nextafter(upper_float, math.inf))
        candidate_count = max(0, upper_order - max(1, lower_order + 1))
        a = max(1, lower_order - 2)
        b = upper_order + 2
        if b > 2**53 - 2:
            raise ValueError("active order exceeds exact float-order limit")
        if candidate_count > max_modes:
            if atol is None:
                raise ValueError("sparse channel exceeds max_modes")
            channel_bounds = _faint_channel_bounds(
                ctx, geometry, a, b, ctx.mpf(float(lo)), n_weights, ctx.mpf(s_depth)
            )
            column = len(ranges)
            for row, bound in enumerate(channel_bounds):
                dense_bounds[row, column] = _outward_float(bound)
            if np.any(dense_bounds[:, column] > atol):
                raise ValueError("dense channel bound exceeds requested atol")
            routes.append("bounded_zero")
        else:
            routes.append("direct")
        ranges.append((a, b))
    cache: dict[int, IntegerAiryEnclosure | IntegerBesselEnclosure] = {}
    method_by_order: dict[int, str] = {}
    depth_i, scale_i = ctx.mpf(depth_ref), ctx.mpf(s_depth)
    light = ctx.mpf(C_SI_M)
    for j, (a, b) in enumerate(ranges):
        if routes[j] == "bounded_zero":
            continue
        for order in range(a, b + 1):
            frequency = order * spacing
            response = _response(ctx, channels, j, frequency)
            if response == zero:
                continue
            if order not in cache:
                airy = None
                if order >= 1000 and z.a >= ctx.mpf("0.9"):
                    try:
                        airy = integer_bessel_airy_enclosure(
                            order, z, decimal_digits=decimal_digits
                        )
                    except ValueError:
                        pass
                if airy is not None and (
                    airy.j_error <= bessel_tolerance
                    and airy.j_prime_error <= bessel_tolerance
                ):
                    cache[order] = airy
                    method_by_order[order] = "airy"
                else:
                    cache[order] = integer_bessel_enclosure(
                        order,
                        z,
                        abs_tolerance=bessel_tolerance,
                        max_nodes=max_bessel_nodes,
                        decimal_digits=decimal_digits,
                    )
                    method_by_order[order] = "contour"
            line = cache[order]
            approximate_j = (
                line.airy_j
                if isinstance(line, IntegerAiryEnclosure)
                else line.trapezoid_j
            )
            approximate_prime = (
                line.airy_j_prime
                if isinstance(line, IntegerAiryEnclosure)
                else line.trapezoid_j_prime
            )

            def line_stokes(j_value, prime):
                return _stokes(
                    ctx,
                    order,
                    geometry,
                    ctx.mpf([j_value.a, j_value.b]),
                    ctx.mpf([prime.a, prime.b]),
                )

            exact_stokes = line_stokes(line.j, line.j_prime)
            approximate_stokes = line_stokes(approximate_j, approximate_prime)
            tau = 2 * (light / frequency) ** 2
            base = ctx.exp(1j * tau * depth_i) if phase is not None else ctx.mpc(1)
            weights = [base]
            for degree in range(1, n_weights):
                weights.append(weights[-1] * (1j * tau * scale_i) / degree)
            full[0][j] += response * exact_stokes[0]
            full[1][j] += response * exact_stokes[2]
            approximate[0][j] += response * approximate_stokes[0]
            approximate[1][j] += response * approximate_stokes[2]
            for degree, weight in enumerate(weights):
                full[2 + degree][j] += response * exact_stokes[1] * weight
                approximate[2 + degree][j] += response * approximate_stokes[1] * weight

    for row in range(2 + n_weights):
        for j in range(channels.n_ch):
            component = _component_error(
                ctx, full[row][j], approximate[row][j], complex_value=row >= 2
            )
            values[row, j] = component[0]
            total_error[row, j] = component[1]
            bessel_error[row, j] = component[2]
            arithmetic_error[row, j] = component[3]
    combined_error = total_error + dense_bounds
    total_error = np.where(
        combined_error > 0, np.nextafter(combined_error, math.inf), 0.0
    )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(total_error)):
        raise ArithmeticError("sparse interval enclosure is not finite")
    if atol is not None and np.any(total_error > atol):
        raise ArithmeticError("sparse interval enclosure exceeds requested atol")
    return SparseChannelEnclosure(
        HostModes(values[0].real, values[1].real, values[2:]),
        tuple(ranges),
        tuple(routes),
        len(cache),
        tuple(sorted(method_by_order.items())),
        ErrorTerm(
            total_error,
            "bound",
            "whole-channel interval or DLMF magnitude bound for exact binary inputs; mpmath interval elementary functions",
            "E_num",
        ),
        ErrorTerm(
            bessel_error,
            "bound",
            "analytic contour or Airy remainder plus enclosing interval spread",
            "E_num",
        ),
        ErrorTerm(
            arithmetic_error,
            "bound",
            "interval arithmetic and final float-midpoint rounding",
            "E_num",
        ),
        ErrorTerm(
            dense_bounds,
            "bound",
            "DLMF 10.14.5--6 whole-range bound for zero-valued dense channels; "
            "all other possibly supported lines included",
            "E_num",
        ),
    )


__all__ = ["SparseChannelEnclosure", "certified_sparse_channel_modes"]
