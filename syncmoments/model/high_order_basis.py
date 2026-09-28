"""Eager low Taylor-order moment bases for the high-order harmonic route.

The expensive harmonic and angular calculations run once here.  Repeated
``predict`` calls use the resulting :class:`~syncmoments.model.basis.SpectralBasis`
matrix. ``Truncation.N`` may be zero or one; first derivatives are evaluated
in physical gamma/B and converted to the reference's dimensionless moment
coordinates. This builder returns an exploratory basis
with ``numerical`` error ``unbounded`` even if the inner quadrature stopping
check succeeds. An explicit optional postprocessor can attach a conditional
coarse fixed-reference numerical envelope, including N=1 derivative columns,
without changing the fast matrix. Displaced-parameter Taylor remainders have
no certificate.

Angular integration uses an even Gauss--Legendre rule in mu.  By default the
inner eta rule follows the narrow high-gamma ridge near eta=mu through the
coordinate eta=mu+h(mu)*sinh(t), with panels split at t=-3 and t=3.  A plain
tensor rule remains available for low-order comparisons.  The symmetry
``(mu, eta) -> (-mu, -eta)`` leaves I and natural-basis Q unchanged and
reverses V, so only positive-mu nodes require harmonic calls.  Angular
quadrature error is not bounded by either rule.
"""

from __future__ import annotations

from collections import Counter
import math

import jax.numpy as jnp
import numpy as np

from ..constants import C_CGS, E_ESU, M_E
from ._basis_checks import UNITS, check_phase, check_reference, truncation_record
from ._basis_core import package_version
from .basis import KernelTerms, SpectralBasis
from .channels import Channels
from .errors import ErrorTerm, Provenance
from .high_order_harmonic import (
    _prepare_host_responses,
    _single_channel,
    high_order_channel_derivatives,
    high_order_channel_modes,
)
from .index import MomentIndex, Truncation
from .moments import Reference, Support
from .phase import TaylorPhase


def _rule_size(value, name, *, even=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{name} must be an integer >= 2")
    if even and value % 2:
        raise ValueError(f"{name} must be even for positive-mu parity reduction")
    return value


def _settings_record(settings):
    return tuple((name, settings[name]) for name in sorted(settings))


def _ridge_eta_rule(mu, nodes, weights, gamma, B, frequency_floor):
    """Gauss panels in t where eta=mu+h*sinh(t), mapped back to [-1, 1].

    At eta=mu, the line spacing is nu_B / (1-beta*mu**2).  Its lowest
    supported harmonic estimates this channel's turning-point width m**(-1/3).
    The physical beaming width gamma**(-1) is the other
    scale.  Multiplication by D=1-beta*mu**2 converts either angular width
    into the local eta-mu width.  This choice sets a quadrature coordinate;
    it is not a support truncation or a quadrature-error certificate.
    """
    beta = math.sqrt(1.0 - gamma**-2)
    D = 1.0 - beta * mu * mu
    spacing = E_ESU * B / (2.0 * math.pi * gamma * M_E * C_CGS * D)
    first_order = max(1.0, frequency_floor / spacing)
    h = D * max(1.0 / gamma, first_order ** (-1.0 / 3.0))
    if not (math.isfinite(h) and h > 0.0):
        raise ValueError("ridge angular scale must be positive and finite")
    t_lo = math.asinh((-1.0 - mu) / h)
    t_hi = math.asinh((1.0 - mu) / h)
    boundaries = (t_lo, *(v for v in (-3.0, 3.0) if t_lo < v < t_hi), t_hi)
    eta_rows, weight_rows = [], []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        midpoint, half = (a + b) / 2.0, (b - a) / 2.0
        t = midpoint + half * nodes
        eta_rows.append(np.clip(mu + h * np.sinh(t), -1.0, 1.0))
        weight_rows.append(half * weights * h * np.cosh(t))
    return np.concatenate(eta_rows), np.concatenate(weight_rows)


def build_high_order_basis(
    channels: Channels,
    truncation: Truncation,
    reference: Reference,
    *,
    support: Support,
    phase: TaylorPhase | None = None,
    n_mu: int = 24,
    n_eta: int = 24,
    angular_rule: str = "ridge",
    direct_limit: int = 4096,
    quad_order: int = 32,
    max_quad_order: int = 2048,
    rtol: float = 1e-6,
    atol: float = 0.0,
    atol_gamma: float = 0.0,
    atol_B: float = 0.0,
    allow_unconverged: bool = False,
    E_phys: ErrorTerm | None = None,
) -> SpectralBasis:
    """Build a reusable high-order ``N<=1`` response at a fixed reference.

    The returned basis retains all angular Legendre pairs and depth-phase
    weights of ``truncation``.  It can be passed to ``predict``; likelihood
    evaluations then perform only matrix algebra.  The full positive harmonic
    range meeting each channel is selected at every angular node, with the
    direct/dense choice made from that channel's active mode count.

    ``angular_rule="ridge"`` uses ``n_eta`` Gauss nodes *per inner panel*;
    the number of panels is one to three for each mu node.  The coordinate
    covers the full eta interval and discards no angular tail.  The local
    scale is ``(1-beta*mu**2)*max(1/gamma,m_lo**(-1/3))``, where ``m_lo`` is
    the channel's lower support frequency divided by its line spacing at
    eta=mu.  Every channel gets its own eta grid: channels at very different
    frequencies can have very different turning-point widths.
    ``angular_rule="tensor"`` uses ``n_eta`` nodes across all of eta.

    At ``N=1``, the gamma and B derivative columns include the line power,
    channel response and Faraday phase derivatives, then multiply by the
    corresponding ``reference.scales``. The derivative evaluator uses
    ``atol_gamma`` and ``atol_B`` as separate *physical-unit* quadrature
    stopping thresholds. They are not error bounds. The finite angular rule
    and host harmonic values/derivatives are not certified by this builder;
    ``kernel_terms.numerical`` is ``unbounded`` until the optional separate
    fixed-reference certificate is requested. Its N=1 derivative bound is
    coarse. No bound for a nonzero gamma/B Taylor displacement is inferred
    from either basis.
    ``allow_unconverged=True`` permits failed inner quadrature-difference
    checks for exploratory work; the count is recorded in provenance.
    ``E_phys`` is an optional *declared* model discrepancy, not a numerical
    certificate.  Only :class:`TaylorPhase` is accepted in this first route.
    """
    if not isinstance(channels, Channels):
        raise ValueError("channels must be a Channels")
    if not isinstance(truncation, Truncation):
        raise ValueError("truncation must be a Truncation")
    if not isinstance(reference, Reference):
        raise ValueError("reference must be a Reference")
    if not isinstance(support, Support):
        raise ValueError("support must be a Support")
    if truncation.N > 1:
        raise ValueError("high-order basis currently requires Truncation.N <= 1")
    phase = TaylorPhase(truncation.max_b()) if phase is None else phase
    if not isinstance(phase, TaylorPhase):
        raise ValueError("high-order basis currently supports TaylorPhase only")
    check_phase(phase, truncation)
    if E_phys is not None and not isinstance(E_phys, ErrorTerm):
        raise ValueError("E_phys must be an ErrorTerm or None")
    n_mu = _rule_size(n_mu, "n_mu", even=True)
    n_eta = _rule_size(n_eta, "n_eta")
    if angular_rule not in ("ridge", "tensor"):
        raise ValueError("angular_rule must be 'ridge' or 'tensor'")
    max_panels = 3 if angular_rule == "ridge" else 1
    if n_mu * n_eta * max_panels * channels.n_ch > 65536:
        raise ValueError("angular evaluation exceeds the 65536-node resource guard")

    notes: list[str] = []
    check_reference(reference, support, notes)
    gamma0, B0 = float(np.asarray(reference.gamma0)), float(np.asarray(reference.B0))
    s_gamma, s_B = (float(np.asarray(v)) for v in reference.scales[:2])
    depth_ref = float(np.asarray(reference.depth_ref))
    s_depth = float(np.asarray(reference.scales[2]))
    index = MomentIndex.build(truncation, components=("I", "Q", "V"))
    present_orders = {(r, s) for _, _, r, s, _ in index.h0 + index.h2}
    derivative_orders = tuple(
        order for order in ((0, 0), (1, 0), (0, 1)) if order in present_orders
    )
    needs_derivatives = len(derivative_orders) > 1
    if needs_derivatives and channels.family != "bump":
        raise ValueError("high-order gamma/B derivatives require bump channels")
    order_lookup = {order: pos for pos, order in enumerate(derivative_orders)}
    mode_positions = tuple(
        ((0, 0), (1, 0), (0, 1)).index(order) for order in derivative_orders
    )
    n_ch, n_weights = channels.n_ch, phase.n_weights
    pairs = np.asarray(index.pairs, dtype=int)
    parity = (-1.0) ** (pairs[:, 0] + pairs[:, 1])
    norm = (2 * pairs[:, 0] + 1) * (2 * pairs[:, 1] + 1) / 4.0

    # The positive half of the full mu rule; the eta nodes may depend on mu.
    mu_all, w_mu_all = np.polynomial.legendre.leggauss(n_mu)
    eta_rule_nodes, eta_rule_weights = np.polynomial.legendre.leggauss(n_eta)
    positive = mu_all > 0
    mu_nodes, w_mu = mu_all[positive], w_mu_all[positive]
    p_mu = np.polynomial.legendre.legvander(mu_nodes, truncation.L_mu)
    channel_responses = tuple(_single_channel(channels, j) for j in range(n_ch))
    frequency_floors = np.asarray(channels.support, dtype=float)[:, 0]

    settings = dict(
        direct_limit=direct_limit,
        quad_order=quad_order,
        max_quad_order=max_quad_order,
        rtol=rtol,
        atol=atol,
        allow_unconverged=allow_unconverged,
    )
    if needs_derivatives:
        settings.update(atol_gamma=atol_gamma, atol_B=atol_B)
    elif atol_gamma != 0.0 or atol_B != 0.0:
        raise ValueError("derivative tolerances require retained gamma/B rows")
    I_projected = np.zeros((len(derivative_orders), n_ch, index.n_lk), dtype=float)
    V_projected = np.zeros_like(I_projected)
    P_projected = np.zeros(
        (len(derivative_orders), n_weights, n_ch, index.n_lk), dtype=complex
    )
    route_counts: Counter[str] = Counter()
    unconverged = 0
    real_order_samples = 0
    angular_evaluations = 0
    order_min, order_max = math.inf, 0
    evaluator = (
        high_order_channel_derivatives
        if needs_derivatives
        else high_order_channel_modes
    )
    for channel, channel_response in enumerate(channel_responses):
        prepared_responses = _prepare_host_responses(
            channel_response, derivatives=needs_derivatives
        )
        for i, mu in enumerate(mu_nodes):
            if angular_rule == "ridge":
                eta_nodes, w_eta = _ridge_eta_rule(
                    float(mu),
                    eta_rule_nodes,
                    eta_rule_weights,
                    gamma0,
                    B0,
                    float(frequency_floors[channel]),
                )
            else:
                eta_nodes, w_eta = eta_rule_nodes, eta_rule_weights
            p_eta = np.polynomial.legendre.legvander(eta_nodes, truncation.L_eta)
            angular = (
                w_mu[i]
                * w_eta[:, None]
                * p_mu[i, pairs[:, 0]][None, :]
                * p_eta[:, pairs[:, 1]]
                * norm[None, :]
            )
            even_weight = angular * (1.0 + parity)[None, :]
            odd_weight = angular * (1.0 - parity)[None, :]
            values_I = np.empty((len(derivative_orders), len(eta_nodes)), dtype=float)
            values_V = np.empty_like(values_I)
            values_P = np.empty(
                (len(derivative_orders), len(eta_nodes), n_weights), dtype=complex
            )
            for k, eta in enumerate(eta_nodes):
                result = evaluator(
                    channel_response,
                    gamma0,
                    B0,
                    float(mu),
                    float(eta),
                    phase=phase,
                    depth_ref=depth_ref,
                    s_depth=s_depth,
                    _prepared_responses=prepared_responses,
                    **settings,
                )
                point_modes = (
                    (result.modes, result.d_gamma, result.d_B)
                    if needs_derivatives
                    else (result.modes,)
                )
                for pos, source in enumerate(mode_positions):
                    modes = point_modes[source]
                    values_I[pos, k] = modes.I[0]
                    values_V[pos, k] = modes.V[0]
                    values_P[pos, k] = modes.P[:, 0]
                angular_evaluations += 1
                route_counts.update(result.routes)
                unconverged += sum(not good for good in result.quadrature_converged)
                real_order_samples += result.evaluated_orders
                for lo, hi in result.active_ranges:
                    if lo <= hi:
                        order_min, order_max = min(order_min, lo), max(order_max, hi)
            I_projected[:, channel] += np.einsum("ak,kp->ap", values_I, even_weight)
            V_projected[:, channel] += np.einsum("ak,kp->ap", values_V, odd_weight)
            P_projected[:, :, channel] += np.einsum(
                "akb,kp->abp", values_P, even_weight
            )

    lookup = {pair: pos for pos, pair in enumerate(index.pairs)}
    scales = {(0, 0): 1.0, (1, 0): s_gamma, (0, 1): s_B}
    I_basis = np.stack(
        [
            I_projected[order_lookup[(r, s)], :, lookup[(l, k)]]
            * scales[(r, s)]
            * ((l + k) % 2 == 0)
            for l, k, r, s, _ in index.h0
        ],
        axis=1,
    )
    V_basis = np.stack(
        [
            V_projected[order_lookup[(r, s)], :, lookup[(l, k)]]
            * scales[(r, s)]
            * ((l + k) % 2 == 1)
            for l, k, r, s, _ in index.h0
        ],
        axis=1,
    )
    P_basis = np.stack(
        [
            P_projected[order_lookup[(r, s)], b, :, lookup[(l, k)]] * scales[(r, s)]
            for l, k, r, s, b in index.h2
        ],
        axis=1,
    )
    if not all(np.all(np.isfinite(array)) for array in (I_basis, V_basis, P_basis)):
        raise ArithmeticError("nonfinite high-order angular basis coefficient")

    unresolved = (
        "host SciPy Bessel, dense harmonic-sum, floating-point, "
        + ("derivative, " if needs_derivatives else "")
        + "and angular quadrature errors have no combined column bound"
    )
    numerical = ErrorTerm.unbounded(unresolved, "E_num")
    terms = KernelTerms(
        physical_kernel=(
            E_phys
            if E_phys is not None
            else ErrorTerm.unbounded(
                "vacuum helical-orbit and pure-rotation physical discrepancy "
                "not supplied",
                "E_phys",
            )
        ),
        harmonic_truncation=ErrorTerm(
            jnp.zeros((n_ch, 4)),
            "bound",
            "full channel-active positive harmonic range evaluated at every "
            "angular node; angular quadrature error is in numerical",
            "E_num",
        ),
        numerical=numerical,
        screen_exponent=ErrorTerm.not_applicable(
            "Taylor phase route: no screen characteristic function is truncated"
        ),
    )
    notes.extend(
        (
            "high-order host basis is exploratory: numerical error unbounded",
            (
                "ridge angular nodes follow eta=mu+h*sinh(t), with full eta "
                "coverage and no certified angular quadrature bound"
                if angular_rule == "ridge"
                else "tensor angular quadrature has no certified error bound"
            ),
            (
                "gamma/B Taylor order is one; derivative coefficients and "
                "displacement remainder are not bounded"
                if needs_derivatives
                else "gamma/B Taylor order is zero; displacement remainder is not bounded"
            ),
            "active_order_envelope is only a minimum/maximum over angular nodes "
            "and channels; real_order_samples counts continuous-order quadrature "
            "samples as well as direct integer lines",
            "incident polarisation not modelled",
        )
    )
    if unconverged:
        notes.append(
            f"{unconverged} angular-node/channel quadrature differences did not "
            "meet their stopping tolerances"
        )
    provenance = Provenance(
        package_version=package_version(),
        kernel=(
            ("name", "high_order_harmonic"),
            ("route", "host_scipy_real_order"),
            ("components", ("I", "Q", "V")),
            ("settings", _settings_record(settings)),
        ),
        channels=tuple(channels.describe()),
        truncation=truncation_record(index),
        reference=tuple(reference.describe()),
        support=tuple(support.describe()),
        phase_route=tuple(phase.describe()),
        quadrature=(
            ("route", f"{angular_rule}_positive_mu_parity"),
            ("n_mu", n_mu),
            ("n_eta", n_eta),
            ("eta_nodes_per_panel", angular_rule == "ridge"),
            ("ridge_panel_splits", (-3.0, 3.0) if angular_rule == "ridge" else ()),
            ("convergence", False),
            ("numerical_route", None),
            ("check_route", None),
        ),
        assumptions=tuple(phase.forced_assumptions()),
        units=UNITS,
        numerics=(
            (
                "active_order_envelope",
                None if order_max == 0 else (int(order_min), int(order_max)),
            ),
            ("real_order_samples", real_order_samples),
            ("routes", tuple(sorted(route_counts.items()))),
            ("angular_evaluations", angular_evaluations),
            ("unconverged_node_channels", unconverged),
        ),
        certified_orders=(),
        finite_checks=(),
        notes=tuple(notes),
    )
    return SpectralBasis(
        index=index,
        truncation=truncation,
        channels=channels,
        reference=reference,
        support=support,
        I_basis=jnp.asarray(I_basis),
        V_basis=jnp.asarray(V_basis),
        P_basis=jnp.asarray(P_basis),
        kernel_terms=terms,
        provenance=provenance,
        certified_orders=(),
        phase=phase,
    )


__all__ = ["build_high_order_basis"]
