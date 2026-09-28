"""Conditional interval geometry for fixed-particle angular boxes.

Each input endpoint and physical constant is interpreted as its exact binary
float value. The resulting enclosures rely on mpmath's experimental interval
elementary functions. This module only encloses geometry and candidate
integer orders. It does not certify angular integration, line powers, or a
population spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from ..constants import C_CGS, E_ESU, M_E

_MAX_EXACT_FLOAT_INT = 2**53 - 2


@dataclass(frozen=True)
class AngularFixedGeometry:
    """Particle constants shared by many angular cells in one interval context."""

    ctx: object
    inverse_gamma: object
    beta: object
    one_minus_beta: object
    omega_b: object
    two_pi: object


@dataclass(frozen=True)
class PreparedChannelSupports:
    """Exact-binary support endpoints reused across many angular cells."""

    ctx: object
    endpoints: tuple[tuple[object, object], ...]


def _binary_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real binary-float scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real binary-float scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real binary-float scalar")
    return result


def _angular_box(ctx, box, name):
    if not isinstance(box, (tuple, list)) or len(box) != 2:
        raise ValueError(f"{name} must be a (lo, hi) pair")
    lo = _binary_float(box[0], f"{name} lower endpoint")
    hi = _binary_float(box[1], f"{name} upper endpoint")
    if not -1 <= lo <= hi <= 1:
        raise ValueError(f"{name} must lie inside [-1, 1] with lo <= hi")
    return ctx.mpf([lo, hi])


def prepare_angular_geometry(ctx, gamma, B) -> AngularFixedGeometry:
    """Precompute exact-binary fixed-particle constants for angular cells.

    ``gamma > 1`` and ``B > 0`` are required. Zero-emissivity cases should be
    handled before building angular cells. Pass the same ``ctx`` to each cell
    operation; a separate context may have a different working precision.
    """
    gamma = _binary_float(gamma, "gamma")
    B = _binary_float(B, "B")
    if gamma <= 1 or B <= 0:
        raise ValueError("requires gamma > 1 and B > 0")
    g = ctx.mpf(gamma)
    field = ctx.mpf(B)
    inverse = 1 / g
    beta = ctx.sqrt((1 - inverse) * (1 + inverse))
    # 1-beta = gamma^-2/(1+beta), without cancellation near gamma >> 1.
    one_minus_beta = inverse**2 / (1 + beta)
    omega_b = ctx.mpf(E_ESU) * field / (g * ctx.mpf(M_E) * ctx.mpf(C_CGS))
    two_pi = 2 * ctx.pi
    if not omega_b.a > 0:
        raise ArithmeticError("cyclotron angular frequency is not positive")
    return AngularFixedGeometry(ctx, inverse, beta, one_minus_beta, omega_b, two_pi)


def angular_box_geometry(ctx, fixed, mu_box, eta_box):
    """Enclose ``(D, delta, perpendicular, z, omega_b, spacing)`` on a box.

    The tuple matches the fixed-point ``_geometry`` used by the internal
    certified harmonic routes. A box touching ``|mu|=1`` or ``|eta|=1`` may
    have ``z.a == 0``; callers that divide by ``z`` must subdivide or use an
    axis-compatible branch. The physical identity

    ``D² - beta²(1-mu²)(1-eta²) = (eta-beta*mu)² + gamma^-2(1-eta²) >= 0``

    proves ``0 <= z <= 1`` throughout every valid box. Intersecting the raw
    interval with ``[0,1]`` therefore preserves every physical point.
    """
    if not isinstance(fixed, AngularFixedGeometry) or fixed.ctx is not ctx:
        raise ValueError("fixed geometry must use the supplied interval context")
    mu = _angular_box(ctx, mu_box, "mu_box")
    eta = _angular_box(ctx, eta_box, "eta_box")
    beta = fixed.beta
    one_minus_beta = fixed.one_minus_beta
    if mu.a >= 0 and eta.a >= 0:
        one_minus_product = (1 - mu) + mu * (1 - eta)
    elif mu.b <= 0 and eta.b <= 0:
        one_minus_product = (1 + mu) + (-mu) * (1 + eta)
    else:
        product = mu * eta
        one_minus_product = ctx.mpf(
            [max(ctx.mpf(0), (1 - product).a), min(ctx.mpf(2), (1 - product).b)]
        )
    D = one_minus_beta + beta * one_minus_product
    delta = (eta - mu) + mu * one_minus_beta
    perpendicular = beta * ctx.sqrt((1 - mu) * (1 + mu))
    viewing_sine = ctx.sqrt((1 - eta) * (1 + eta))
    if not D.a > 0:
        raise ArithmeticError("angular box has nonpositive Doppler denominator")
    raw_z = perpendicular * viewing_sine / D
    z_lo = max(ctx.mpf(0), raw_z.a)
    z_hi = min(ctx.mpf(1), raw_z.b)
    # Near eta=mu the direct quotient loses the numerator/denominator
    # correlation. The positive defect has a much narrower interval there.
    # Both formulas enclose the same physical z, so their intersection does.
    defect = (delta / D) ** 2 + (fixed.inverse_gamma * viewing_sine / D) ** 2
    if defect.a > 1:
        raise ArithmeticError("angular box has an inconsistent z defect")
    z_hi = min(z_hi, ctx.sqrt(1 - max(ctx.mpf(0), defect.a)).b)
    if defect.b <= 1:
        defect_z = ctx.sqrt(1 - defect)
        z_lo = max(z_lo, defect_z.a)
    if z_lo > z_hi:
        raise ArithmeticError("angular box has an inconsistent z enclosure")
    z = ctx.mpf([z_lo, z_hi])
    spacing = fixed.omega_b / (fixed.two_pi * D)
    if not spacing.a > 0:
        raise ArithmeticError("angular box has nonpositive harmonic spacing")
    return D, delta, perpendicular, z, fixed.omega_b, spacing


def _support_pair(pair):
    if not hasattr(pair, "__len__") or len(pair) != 2:
        raise ValueError("channel support entries must be (lo, hi) pairs")
    lo = _binary_float(pair[0], "channel support lower endpoint")
    hi = _binary_float(pair[1], "channel support upper endpoint")
    if not 0 < lo < hi:
        raise ValueError("channel support requires 0 < lo < hi")
    return lo, hi


def prepare_channel_supports(ctx, supports) -> PreparedChannelSupports:
    """Convert and validate channel endpoints once before angular subdivision."""
    endpoints = tuple(
        (ctx.mpf(lo), ctx.mpf(hi)) for lo, hi in map(_support_pair, supports)
    )
    return PreparedChannelSupports(ctx, endpoints)


def active_channel_ranges(ctx, supports, spacing, *, padding=2):
    """Conservative per-channel union of integers active anywhere in a cell.

    For ``spacing in [s_lo,s_hi]`` and open support ``(nu_lo,nu_hi)``, every
    possible active integer lies between ``nu_lo/s_hi`` and ``nu_hi/s_lo``.
    Conversion to binary64 is followed by outward ``nextafter`` and integer
    padding. The guarded limit ensures this conversion cannot skip more than
    one integer. Returns ``(0,-1)`` only when the entire support is below the
    first harmonic. No array of the selected mode integers is allocated.
    """
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 2:
        raise ValueError("padding must be an integer >= 2")
    spacing = ctx.mpf(spacing)
    if not spacing.a > 0:
        raise ValueError("spacing must be a positive interval")
    if isinstance(supports, PreparedChannelSupports):
        if supports.ctx is not ctx:
            raise ValueError("prepared supports must use the supplied interval context")
        endpoints = supports.endpoints
    else:
        endpoints = prepare_channel_supports(ctx, supports).endpoints
    ranges = []
    for lo, hi in endpoints:
        if (hi / spacing).b < 1:
            ranges.append((0, -1))
            continue
        lower = lo / spacing
        upper = hi / spacing
        lower_float = float(lower.a)
        upper_float = float(upper.b)
        if not (math.isfinite(lower_float) and math.isfinite(upper_float)):
            raise ValueError("active harmonic order is not representable")
        first = max(1, math.floor(math.nextafter(lower_float, -math.inf)) - padding)
        last = math.ceil(math.nextafter(upper_float, math.inf)) + padding
        if last > _MAX_EXACT_FLOAT_INT:
            raise ValueError("active order exceeds exact float-order limit")
        ranges.append((first, last))
    return tuple(ranges)


__all__ = [
    "AngularFixedGeometry",
    "PreparedChannelSupports",
    "prepare_angular_geometry",
    "angular_box_geometry",
    "prepare_channel_supports",
    "active_channel_ranges",
]
