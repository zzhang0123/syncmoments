"""Conditional whole-range magnitude bounds for guarded angular cells.

For fixed exact-binary gamma and B, this internal helper bounds the channel
sum at every point of one angular box. It uses DLMF 10.14.5--6 for the exact
integer Bessel functions and the interval geometry from
``_interval_angular_geometry``. The result remains conditional on mpmath's
experimental interval elementary functions; it is not an angular integral,
population, or physical-model certificate.

The caller must supply a channel response already verified to vanish outside
its stored open support and obey ``0 <= R(nu) <= 1`` inside, such as the
validated unit-peak bump. This helper receives only the lower support
frequency and cannot check that response precondition. A positive bound
smaller than the least positive
binary64 value remains positive as an MP interval; publication as a float
requires outward rounding (for example, toward positive infinity).
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
class FaintBoundConstants:
    """Physical constants converted once for repeated cell evaluations."""

    ctx: object
    e2_over_2pi_c: object
    c_si: object


@dataclass(frozen=True)
class PreparedFaintCell:
    """Guarded geometry extrema reused across channels of one angular cell."""

    ctx: object
    c_si: object
    z_hi: object
    root_lower: object
    xi_lower: object
    prefactor_base_upper: object
    parallel_base_upper: object
    derivative_base_upper: object


def prepare_faint_bound_constants(ctx) -> FaintBoundConstants:
    """Prepare interval constants in the caller's existing precision context."""
    charge = ctx.mpf(E_ESU)
    coefficient = charge**2 / (2 * ctx.pi * ctx.mpf(C_CGS))
    return FaintBoundConstants(ctx, coefficient, ctx.mpf(C_SI_M))


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


def _upper_magnitude(ctx, value):
    """Return a nonnegative upper endpoint, rejecting unusable float scale."""
    upper = value.b
    if not upper >= 0 or not math.isfinite(float(upper)):
        raise ArithmeticError("faint-channel bound is not finite or representable")
    return ctx.mpf([0, upper])


def prepare_faint_cell(
    ctx,
    geometry,
    *,
    constants=None,
    fixed: AngularFixedGeometry | None = None,
    mu_abs_lower=None,
) -> PreparedFaintCell:
    """Precompute guarded geometry; mu_abs_lower must bound |mu| in the box."""
    if constants is None:
        constants = prepare_faint_bound_constants(ctx)
    elif not isinstance(constants, FaintBoundConstants) or constants.ctx is not ctx:
        raise ValueError("faint constants must use the supplied interval context")
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
        and z.b < 1
        and omega_b.a > 0
        and spacing.a > 0
    ):
        raise ValueError("faint route needs positive geometry and 0 <= z < 1")
    z_hi = z.b
    if z_hi == 0:
        zero = ctx.mpf(0)
        return PreparedFaintCell(
            ctx, constants.c_si, zero, zero, zero, zero, zero, zero
        )
    root = ctx.sqrt((1 - z_hi) * (1 + z_hi))
    xi = ctx.ln((1 + root) / z_hi) - root
    xi_lower = max(ctx.mpf(0), xi.a)
    delta_upper = max(abs(delta.a), abs(delta.b))
    p_over_d3 = (perpendicular.b**2 / D.a**3).b
    if fixed is not None:
        # With t=1-|mu|, D>=a+beta*t and p²<=2*beta²*t.
        # The universal maximum of p²/D³ is 8*beta/(27*a²).
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
    # The physical Doppler identity gives |delta|/D <= sqrt(1-z²) <= 1;
    # keep that correlation when an interval ratio is wider near an axis.
    defect_upper = ctx.sqrt((1 - z.a) * (1 + z.a)).b
    delta_ratio_upper = min((delta_upper / D.a).b, defect_upper)
    parallel_base = (delta_ratio_upper / z_hi).b
    derivative_base = (
        ctx.sqrt(ctx.sqrt(1 + z_hi**2)) / (z_hi * ctx.sqrt(2 * ctx.pi))
    ).b
    prefactor_base = (constants.e2_over_2pi_c * omega_b.b**2 * p_over_d3).b
    return PreparedFaintCell(
        ctx,
        constants.c_si,
        z_hi,
        root.a,
        xi_lower,
        prefactor_base,
        parallel_base,
        derivative_base,
    )


def angular_faint_magnitude_bounds(
    ctx,
    geometry,
    first: int,
    last: int,
    frequency_lo,
    *,
    phase: TaylorPhase | None = None,
    s_depth=1.0,
    constants: FaintBoundConstants | None = None,
):
    """Upper ``|I|,|V|,|P_b|`` for all active integers in a guarded box.

    Returns nonnegative mpmath intervals ``[0,U]`` in ``I,V,P_b`` row order.
    A separately validated response that vanishes outside its stored open
    support and has supremum at most one inside is required. The
    exact finite-gamma helical-orbit line powers use
    ``A_parallel=delta*perp*J_m(mz)/(D*z)`` and
    ``A_perp=perp*J'_m(mz)``. With ``xi(z)=log((1+sqrt(1-z²))/z)-sqrt(1-z²)``,
    DLMF `10.14.5 <https://dlmf.nist.gov/10.14.E5>`_ and
    `10.14.6 <https://dlmf.nist.gov/10.14.E6>`_ give
    ``|J_m(mz)|<=exp(-m*xi)`` and
    ``|J'_m(mz)|<=(1+z²)^(1/4)*exp(-m*xi)/(z*sqrt(2*pi*m))``.

    Since ``xi'(z)=-sqrt(1-z²)/z``, the log derivative of
    ``exp(-m*xi(z))/z`` is ``(m*sqrt(1-z²)-1)/z``. The derivative envelope
    adds ``z/(2*(1+z²))`` to this nonnegative derivative. Thus both increase
    on ``0<z<=z_hi`` if ``first*sqrt(1-z_hi²)>1``; both tend to zero at
    ``z=0`` for ``m>=3``. Their maxima occur at ``z_hi`` and ``m=first``.
    The per-line prefactor is bounded with ``m=last`` and positive geometry
    endpoints, then multiplied by the integer count. ``|Q|,|V|<=I`` and
    ``|w_b|<= [2*(c/nu_lo)^2*s_depth]^b/b!`` give the other rows.

    When exponential decay is strictly below one, a closed-form infinite
    geometric second-moment tail may replace the coarser
    ``count*last**2`` factor. Both bounds include every active integer.

    The guarded route requires ``first>=3``, ``z_hi<1`` and the displayed
    monotonicity condition. It may be loose close to the turning point. Pass a
    ``PreparedFaintCell`` as ``geometry`` to reuse its mpmath work across
    channels. It rejects an
    unsupported phase, malformed geometry, nonpositive support or scale, or
    an unrepresentable output. ``depth_ref`` is absent because its pure phase
    has unit modulus. No integer mode array or Bessel call is made.
    """
    if (
        isinstance(first, bool)
        or isinstance(last, bool)
        or not isinstance(first, int)
        or not isinstance(last, int)
        or first < 3
        or last < first
        or last > _MAX_EXACT_FLOAT_INT
    ):
        raise ValueError("faint route requires 3 <= first <= last <= 2**53-2")
    if phase is not None and not isinstance(phase, TaylorPhase):
        raise ValueError("faint route supports TaylorPhase or no phase")
    frequency_lo = _positive_binary_float(frequency_lo, "frequency_lo")
    s_depth = _positive_binary_float(s_depth, "s_depth")
    if isinstance(geometry, PreparedFaintCell):
        if geometry.ctx is not ctx or constants is not None:
            raise ValueError(
                "prepared cell needs its original context and no constants"
            )
        cell = geometry
    else:
        cell = prepare_faint_cell(ctx, geometry, constants=constants)
    n_weights = 1 if phase is None else phase.n_weights
    if cell.z_hi == 0:
        return tuple(ctx.mpf(0) for _ in range(2 + n_weights))
    if not (first * cell.root_lower > 1):
        raise ValueError("faint route monotonicity guard failed")
    decay = ctx.exp(-first * cell.xi_lower).b
    parallel = (cell.parallel_base_upper * decay).b
    transverse = (cell.derivative_base_upper * decay / ctx.sqrt(first)).b
    intensity = (
        (last - first + 1)
        * cell.prefactor_base_upper
        * last**2
        * (parallel**2 + transverse**2)
    ).b
    # With q=exp(-2*xi_lower)<1, summing every m beyond first can be
    # sharper than count*last²*q**first, especially for a very broad active
    # range. The infinite tail safely includes the finite range:
    # sum_{m=n}^infinity m² q^m
    # = q^n [n²/(1-q)+2nq/(1-q)²+q(1+q)/(1-q)³].
    # The derivative envelope contributes an extra 1/m <= 1/first.
    if cell.xi_lower > 0:
        q = ctx.exp(-2 * cell.xi_lower).b
        gap = 1 - q
        if gap.a > 0:
            moment_tail = (
                q**first
                * (first**2 / gap + 2 * first * q / gap**2 + q * (1 + q) / gap**3)
            ).b
            geometric = (
                cell.prefactor_base_upper
                * (cell.parallel_base_upper**2 + cell.derivative_base_upper**2 / first)
                * moment_tail
            ).b
            intensity = min(intensity, geometric)
    bounds = [_upper_magnitude(ctx, intensity), _upper_magnitude(ctx, intensity)]
    tau_max = (2 * (cell.c_si / ctx.mpf(frequency_lo)) ** 2).b
    weight = ctx.mpf(1)
    scale = ctx.mpf(s_depth)
    for degree in range(n_weights):
        if degree:
            weight = (weight * tau_max * scale / degree).b
        bounds.append(_upper_magnitude(ctx, intensity * weight))
    return tuple(bounds)


__all__ = [
    "FaintBoundConstants",
    "PreparedFaintCell",
    "prepare_faint_bound_constants",
    "prepare_faint_cell",
    "angular_faint_magnitude_bounds",
]
