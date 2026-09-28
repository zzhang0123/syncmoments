"""Conditional all-z magnitude envelope for an angular cell's integer lines.

This deliberately coarse envelope covers the exact finite-gamma helical-orbit
reference at every ``0 <= z <= 1``, including axes and the turning region.
The caller must validate a unit-peak bump, or otherwise prove that ``R=0``
outside the stored open support and ``0 <= R <= 1`` inside. Only its positive lower frequency is passed
here. Mpmath's experimental interval elementary functions remain a
conditional numerical assumption. The output is not an angular integral,
population, or physical-model certificate.

Positive MP bounds below binary64's smallest positive number must be rounded
outward if published as floats; an ordinary ``float(bound.b)`` can become
zero and is not a valid upper bound in that case.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from ..constants import C_CGS, C_SI_M, E_ESU
from ._interval_angular_geometry import AngularFixedGeometry
from .phase import TaylorPhase

_MAX_EXACT_FLOAT_INT = 2**53 - 2


@dataclass(frozen=True)
class GlobalBoundConstants:
    """Exact-binary physical constants converted once per interval context."""

    ctx: object
    e2_over_2pi_c: object
    c_si: object
    parallel_order_constant: object
    derivative_order_constant: object
    derivative_two_thirds_constant: object


@dataclass(frozen=True)
class PreparedGlobalCell:
    """Channel-independent whole-range line prefactor bound for one cell."""

    ctx: object
    c_si: object
    intensity_base_upper: object
    parallel_base_upper: object
    transverse_base_upper: object
    parallel_order_constant: object
    derivative_order_constant: object
    derivative_two_thirds_constant: object


def prepare_global_bound_constants(ctx) -> GlobalBoundConstants:
    """Convert fixed physical constants in an existing mpmath context."""
    charge = ctx.mpf(E_ESU)
    coefficient = charge**2 / (2 * ctx.pi * ctx.mpf(C_CGS))
    third = ctx.mpf(1) / 3
    airy_peak = ctx.exp(third * ctx.ln(2)) / (
        ctx.exp(2 * third * ctx.ln(3)) * ctx.gamma(2 * third)
    )
    parallel_constant = (ctx.exp(1) * airy_peak).b
    derivative_constant = (
        ctx.exp(1) * ctx.sqrt(ctx.sqrt(2)) / (2 * ctx.sqrt(2 * ctx.pi))
    ).b
    derivative_two_thirds = (3 * ctx.exp(2 * third * ctx.ln(4)) / (2 * ctx.pi)).b
    return GlobalBoundConstants(
        ctx,
        coefficient,
        ctx.mpf(C_SI_M),
        parallel_constant,
        derivative_constant,
        derivative_two_thirds,
    )


def _positive_binary_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive real scalar") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return result


def _finite_upper(ctx, value):
    upper = value.b
    if not upper >= 0 or not math.isfinite(float(upper)):
        raise ArithmeticError("global angular bound is not finite or representable")
    return ctx.mpf([0, upper])


def prepare_global_cell(
    ctx,
    geometry,
    *,
    constants=None,
    fixed: AngularFixedGeometry | None = None,
    mu_abs_lower=None,
) -> PreparedGlobalCell:
    """Bound the line prefactor once for reuse over channels in one cell.

    If supplied, mu_abs_lower must be a proved lower bound on |mu| throughout
    this geometry. It is used only with matching fixed gamma/B constants.
    """
    if constants is None:
        constants = prepare_global_bound_constants(ctx)
    elif not isinstance(constants, GlobalBoundConstants) or constants.ctx is not ctx:
        raise ValueError("global constants must use the supplied interval context")
    if not isinstance(geometry, tuple) or len(geometry) != 6:
        raise ValueError("geometry must be a six-component interval tuple")
    if fixed is not None and (
        not isinstance(fixed, AngularFixedGeometry) or fixed.ctx is not ctx
    ):
        raise ValueError("fixed geometry must use the supplied interval context")
    if mu_abs_lower is not None:
        if (
            fixed is None
            or isinstance(mu_abs_lower, bool)
            or not isinstance(mu_abs_lower, Real)
        ):
            raise ValueError("mu_abs_lower requires fixed geometry and [0,1]")
        mu_abs_lower = float(mu_abs_lower)
        if not math.isfinite(mu_abs_lower) or not 0 <= mu_abs_lower <= 1:
            raise ValueError("mu_abs_lower requires fixed geometry and [0,1]")
    D, delta, perpendicular, z, omega_b, spacing = geometry
    if not (
        D.a > 0
        and perpendicular.a >= 0
        and z.a >= 0
        and z.b <= 1
        and omega_b.a > 0
        and spacing.a > 0
    ):
        raise ValueError("global route needs positive geometry and 0 <= z <= 1")
    if perpendicular.b == 0:
        zero = ctx.mpf(0)
        return PreparedGlobalCell(
            ctx,
            constants.c_si,
            zero,
            zero,
            zero,
            constants.parallel_order_constant,
            constants.derivative_order_constant,
            constants.derivative_two_thirds_constant,
        )
    delta_upper = max(abs(delta.a), abs(delta.b))
    ratio_upper = (delta_upper / D.a).b
    # D² - beta²(1-mu²)(1-eta²)
    #   = (eta-beta*mu)² + gamma^-2(1-eta²) >= 0.
    # The same identity gives |delta|/D <= sqrt(1-z²), which is at most
    # one. Keep that correlation when independent interval division is
    # broad near an axis or beaming ridge.
    defect_upper = ctx.sqrt((1 - z.a) * (1 + z.a)).b
    ratio_upper = min(ratio_upper, defect_upper)
    ratio_factor = (1 + ratio_upper**2).b
    p_over_d3 = (perpendicular.b**2 / D.a**3).b
    if fixed is not None:
        # For every mu, t=1-|mu| >=0 and |eta|<=1,
        # D>=a+beta*t, p²<=2*beta²*t, a=1-beta>0.
        # The maximum of 2*beta²*t/(a+beta*t)^3 over t>=0
        # is 8*beta/(27*a²), attained at t=a/(2*beta).
        physical_upper = (8 * fixed.beta / (27 * fixed.one_minus_beta**2)).b
        if mu_abs_lower is not None:
            T = 1 - ctx.mpf(mu_abs_lower)
            turning_t = fixed.one_minus_beta / (2 * fixed.beta)
            if T.b <= turning_t.a:
                axis_upper = (
                    2 * fixed.beta**2 * T / (fixed.one_minus_beta + fixed.beta * T) ** 3
                ).b
                physical_upper = min(physical_upper, axis_upper)
        p_over_d3 = min(p_over_d3, physical_upper)
    transverse_base = (constants.e2_over_2pi_c * omega_b.b**2 * p_over_d3).b
    parallel_base = (transverse_base * ratio_upper**2).b
    base = (transverse_base * ratio_factor).b
    if not base >= 0 or not math.isfinite(float(base)):
        raise ArithmeticError("global angular prefactor is not representable")
    return PreparedGlobalCell(
        ctx,
        constants.c_si,
        base,
        parallel_base,
        transverse_base,
        constants.parallel_order_constant,
        constants.derivative_order_constant,
        constants.derivative_two_thirds_constant,
    )


def angular_global_magnitude_bounds(
    ctx,
    geometry,
    first: int,
    last: int,
    frequency_lo,
    *,
    phase: TaylorPhase | None = None,
    s_depth=1.0,
    constants: GlobalBoundConstants | None = None,
):
    """Return MP ``[0,U]`` bounds for ``|I|,|V|,|P_b|`` on every angle.

    DLMF `10.9.2 <https://dlmf.nist.gov/10.9.E2>`_ gives the real-angle
    integer-Bessel integral, hence ``|J_n(x)|<=1``. Differentiating under
    that finite integral gives ``|J'_m(x)|<=1``. DLMF
    `10.6.1 <https://dlmf.nist.gov/10.6.E1>`_ gives
    ``J_m(mz)/z=(J_(m-1)(mz)+J_(m+1)(mz))/2`` for ``z>0``; its continuous
    value at ``z=0`` also has magnitude at most one. Therefore
    ``|A_parallel|<=|delta|*perpendicular/D`` and
    ``|A_perpendicular|<=perpendicular``. DLMF 10.14.2, 10.14.6 and
    10.14.7 additionally give all-z order bounds proportional to
    ``m**(-1/3)`` for ``J_m(mz)/z`` and ``m**(-1/2)`` for ``J'_m(mz)``.
    Splitting the DLMF 10.9.2 derivative integral at
    ``delta=(4/m)**(1/3)`` and integrating the outer part by parts gives
    the stronger uniform ``C*m**(-2/3)`` derivative bound, where
    ``C=3*4**(2/3)/(2*pi)``; the direct integral handles ``m=1``.
    For each term use the smallest available bound; after
    multiplication by ``m**2`` both alternatives increase with order,
    so their maximum is at ``last``. Multiplying by the inclusive integer
    count, using
    ``|Q|,|V|<=I`` and the Taylor weight magnitude gives the returned rows.

    ``geometry`` may be a six-component interval tuple or a
    ``PreparedGlobalCell`` reused across channels. The caller supplies a
    conservative active range and a response that is zero outside the stored
    open support and has proven supremum at most one inside. The empty range
    sentinel ``(0,-1)`` yields zero after geometry
    validation. No mode array or Bessel evaluation is performed.
    """
    if (
        isinstance(first, bool)
        or isinstance(last, bool)
        or not isinstance(first, int)
        or not isinstance(last, int)
    ):
        raise ValueError("global route requires integer first and last")
    empty = first == 0 and last == -1
    if not empty and (first < 1 or last < first or last > _MAX_EXACT_FLOAT_INT):
        raise ValueError("global route requires 1 <= first <= last <= 2**53-2")
    if phase is not None and not isinstance(phase, TaylorPhase):
        raise ValueError("global route supports TaylorPhase or no phase")
    frequency_lo = _positive_binary_float(frequency_lo, "frequency_lo")
    s_depth = _positive_binary_float(s_depth, "s_depth")
    if isinstance(geometry, PreparedGlobalCell):
        if geometry.ctx is not ctx or constants is not None:
            raise ValueError(
                "prepared cell needs its original context and no constants"
            )
        cell = geometry
    else:
        cell = prepare_global_cell(ctx, geometry, constants=constants)
    n_weights = 1 if phase is None else phase.n_weights
    if empty or cell.intensity_base_upper == 0:
        return tuple(ctx.mpf(0) for _ in range(2 + n_weights))
    count = last - first + 1
    old_intensity = (count * last**2 * cell.intensity_base_upper).b
    # DLMF 10.14.E2 and 10.14.E7 bound |J_m(mz)/z| by
    # e*C_A*m^(-1/3), C_A=2^(1/3)/(3^(2/3)*Gamma(2/3)).
    # The factor z^(m-1)*exp(m*(1-z)) is at most e on [0,1].
    # DLMF 10.14.E6 gives |J'_m(mz)| <= K*m^(-1/2), where
    # exp(-m*xi(z))/z <= exp(-xi(z))/z <= e/2 and
    # K=e*2^(1/4)/(2*sqrt(2*pi)). Limits cover z=0.
    # https://dlmf.nist.gov/10.14.E2
    # https://dlmf.nist.gov/10.14.E6
    # https://dlmf.nist.gov/10.14.E7
    order = ctx.mpf(last)
    order_cube_root = ctx.exp(ctx.ln(order) / 3).b
    parallel_order = min(
        ctx.mpf(last**2),
        (cell.parallel_order_constant**2 * order * order_cube_root).b,
    )
    # A second bound from the DLMF 10.9.E2 integer real-angle integral:
    # split the derivative integral at delta=(4/m)^(1/3). On [0,delta],
    # the absolute integral is <=delta²/(2*pi). On [delta,pi],
    # f=sin(t)/(1-z*cos(t)) is unimodal and sup(f)<=cot(delta/2)<=2/delta.
    # Integration by parts then gives <=4/(pi*m*delta), hence
    # |J'_m(mz)|<=C*m^(-2/3), C=3*4^(2/3)/(2*pi), m>=2.
    # At m=1 the direct integral gives 2/pi<C. This covers 0<=z<=1.
    # https://dlmf.nist.gov/10.9.E2
    transverse_order = min(
        ctx.mpf(last**2),
        (cell.derivative_order_constant**2 * order).b,
        (cell.derivative_two_thirds_constant**2 * order_cube_root**2).b,
    )
    scaled_intensity = (
        count
        * (
            cell.parallel_base_upper * parallel_order
            + cell.transverse_base_upper * transverse_order
        )
    ).b
    intensity = min(old_intensity, scaled_intensity)
    bound = _finite_upper(ctx, intensity)
    bounds = [bound, bound, bound]
    if n_weights > 1:
        tau_max = (2 * (cell.c_si / ctx.mpf(frequency_lo)) ** 2).b
        scale = ctx.mpf(s_depth)
        weight = ctx.mpf(1)
        for degree in range(1, n_weights):
            weight = (weight * tau_max * scale / degree).b
            bounds.append(_finite_upper(ctx, intensity * weight))
    return tuple(bounds)


def angular_global_first_derivative_bounds(
    ctx,
    geometry,
    fixed: AngularFixedGeometry,
    first: int,
    last: int,
    frequency_lo,
    frequency_hi,
    width,
    B,
    mu_abs_upper,
    eta_abs_upper,
    *,
    depth_ref,
    s_depth,
    max_b: int,
    constants: GlobalBoundConstants,
):
    """Whole-cell magnitudes of the physical gamma/B derivatives.

    The two returned tuples have entries for the unpolarised I/V line sum,
    then P depth weights b=0..max_b. They bound the derivative of the exact
    *integer* harmonic sum at fixed angular coordinates. The caller must
    prove the ideal unit-peak bump is smooth at its stored support edges.

    `DLMF 10.9.2 <https://dlmf.nist.gov/10.9.E2>`_ gives
    |J_n(x)|, |J'_n(x)| <= 1 for every nonnegative integer
    n and real x; the second follows by differentiating the finite cosine
    integral. Hence the neighbouring-order amplitudes and their gamma
    derivatives are bounded without evaluating a Bessel function. For V,
    Cauchy--Schwarz bounds the cross term by the product of the amplitude
    and derivative vector norms. For the bump, put u=1/(1-t**2)>=1:
    |dR/dnu| = 2*|t|*u**2*exp(1-u)/width, and the maximum of
    u**2*exp(1-u) is 4/e at u=2. Thus |dR/dnu|<=8/(e*width).
    This route is intentionally coarse.
    """
    if not isinstance(fixed, AngularFixedGeometry) or fixed.ctx is not ctx:
        raise ValueError("fixed geometry must use the supplied interval context")
    if not isinstance(constants, GlobalBoundConstants) or constants.ctx is not ctx:
        raise ValueError("global constants must use the supplied interval context")
    if (
        isinstance(first, bool)
        or isinstance(last, bool)
        or not isinstance(first, int)
        or not isinstance(last, int)
        or (first, last) != (0, -1)
        and (first < 1 or last < first or last > _MAX_EXACT_FLOAT_INT)
    ):
        raise ValueError("derivative global route requires an active integer range")
    if isinstance(max_b, bool) or not isinstance(max_b, int) or max_b < 0:
        raise ValueError("max_b must be a nonnegative integer")
    if len(geometry) != 6:
        raise ValueError("geometry must be a six-component interval tuple")
    zero = ctx.mpf(0)
    if first == 0:
        return tuple(zero for _ in range(max_b + 2)), tuple(
            zero for _ in range(max_b + 2)
        )
    D, delta, perpendicular, z, omega_b, _ = geometry
    if not (D.a > 0 and perpendicular.a >= 0 and z.a >= 0 and z.b <= 1):
        raise ValueError("derivative global route needs positive physical geometry")
    lo, hi, band_width, field = (
        ctx.mpf(v) for v in (frequency_lo, frequency_hi, width, B)
    )
    mu_u, eta_u = (ctx.mpf(v) for v in (mu_abs_upper, eta_abs_upper))
    depth = ctx.mpf(depth_ref)
    scale = ctx.mpf(s_depth)
    if not (
        lo.a > 0
        and hi.a > lo.b
        and band_width.a > 0
        and field.a > 0
        and 0 <= mu_u.a <= mu_u.b <= 1
        and 0 <= eta_u.a <= eta_u.b <= 1
        and scale.a > 0
    ):
        raise ValueError("invalid derivative band, angle or depth envelope")
    beta = fixed.beta
    gamma_inv = fixed.inverse_gamma
    beta_gamma = (gamma_inv**3 / beta).b
    mueta = (mu_u.b * eta_u.b).b
    delta_u = max(abs(delta.a), abs(delta.b))
    p_u, D_lo, z_u = perpendicular.b, D.a, z.b
    A_parallel = (delta_u * p_u / D_lo).b
    A_perp = p_u
    z_gamma = (z_u * beta_gamma * (1 / beta.a + mueta / D_lo)).b
    A_parallel_gamma = (
        beta_gamma * p_u * (mu_u.b + delta_u / beta.a) / D_lo
        + delta_u * p_u * beta_gamma * mueta / D_lo**2
        + last * A_parallel * z_gamma
    ).b
    A_perp_gamma = (p_u * beta_gamma / beta.a + last * p_u * z_gamma).b
    amplitude_squared = (A_parallel**2 + A_perp**2).b
    amplitude = ctx.sqrt(amplitude_squared).b
    derivative_amplitude = ctx.sqrt(A_parallel_gamma**2 + A_perp_gamma**2).b
    prefactor = (constants.e2_over_2pi_c * omega_b.b**2 * last**2 / D_lo**3).b
    kappa = (2 * gamma_inv.b + 3 * beta_gamma * mueta / D_lo).b
    line_gamma = (
        (last - first + 1)
        * prefactor
        * (kappa * amplitude_squared + 2 * amplitude * derivative_amplitude)
    ).b
    # A response-weighted intensity upper cannot bound R' near a compact
    # edge. Use the unweighted positive line-intensity envelope instead.
    unweighted = angular_global_magnitude_bounds(
        ctx, geometry, first, last, frequency_lo, constants=constants
    )[0].b
    response_slope = (8 / (ctx.exp(1).a * band_width.a)).b
    moving_response = (hi.b * response_slope).b
    frequency_gamma = (gamma_inv.b + beta_gamma * mueta / D_lo).b
    tau_max = (2 * (constants.c_si / lo.a) ** 2).b
    abs_depth = max(abs(depth.a), abs(depth.b))
    gamma_rows = [
        _finite_upper(ctx, line_gamma + unweighted * frequency_gamma * moving_response)
    ]
    B_rows = [_finite_upper(ctx, unweighted * (2 + moving_response) / field.a)]
    weight = ctx.mpf(1)
    for degree in range(max_b + 1):
        if degree:
            weight = (weight * tau_max * scale.b / degree).b
        phase_slope = (2 * (tau_max * abs_depth + degree)).b
        gamma_rows.append(
            _finite_upper(
                ctx,
                weight
                * (
                    line_gamma
                    + unweighted * frequency_gamma * (moving_response + phase_slope)
                ),
            )
        )
        B_rows.append(
            _finite_upper(
                ctx,
                weight * unweighted * (2 + moving_response + phase_slope) / field.a,
            )
        )
    return tuple(gamma_rows), tuple(B_rows)


__all__ = [
    "GlobalBoundConstants",
    "PreparedGlobalCell",
    "prepare_global_bound_constants",
    "prepare_global_cell",
    "angular_global_magnitude_bounds",
    "angular_global_first_derivative_bounds",
]
