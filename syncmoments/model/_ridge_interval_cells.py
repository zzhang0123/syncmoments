"""Conditional interval cells for a two-patch correlated angular map.

For a fixed positive-mu slab, each side maps ``u in [0,1]`` onto one side of
the viewing-angle domain. Intervals preserve the mu/eta correlation near the
beaming ridge. This is geometry only: mpmath's experimental interval
elementary functions remain an assumption, and no angular integral or
population error is certified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._interval_angular_geometry import AngularFixedGeometry, _binary_float


@dataclass(frozen=True)
class RidgeSlab:
    """Exact-binary positive-mu slab and reusable map endpoint values."""

    ctx: object
    fixed: AngularFixedGeometry
    mu_lo: float
    mu_hi: float
    h: float
    L_plus_lo: object
    L_plus_hi: object
    L_minus_lo: object
    L_minus_hi: object


@dataclass(frozen=True)
class RidgeIntervalCell:
    """Geometry, viewing angle and nonnegative absolute eta Jacobian hulls."""

    geometry: tuple
    eta: object
    jacobian: object


def _asinh_nonnegative(ctx, x):
    """asinh(x) for x>=0 using log1p to preserve small positive x."""
    root = ctx.sqrt(1 + x**2)
    return ctx.log1p(x + x**2 / (1 + root))


def _sinh_nonnegative(ctx, x):
    """sinh(x) for x>=0 without subtracting nearly equal exponentials."""
    return ctx.expm1(x) * (1 + ctx.exp(-x)) / 2


def _cosh_nonnegative(ctx, x):
    return (ctx.exp(x) + ctx.exp(-x)) / 2


def prepare_ridge_slab(ctx, fixed, mu_box, *, h=None) -> RidgeSlab:
    """Prepare a positive-mu slab; default ``h`` is rounded ``1/gamma``.

    Any positive exact binary ``h`` gives the same complete angle coverage;
    it only changes how the ``u`` coordinate concentrates near ``eta=mu``.
    Endpoint values of ``L_s(mu)=asinh((1-s*mu)/h)`` are reused by cells.
    """
    if not isinstance(fixed, AngularFixedGeometry) or fixed.ctx is not ctx:
        raise ValueError("fixed geometry must use the supplied interval context")
    if not isinstance(mu_box, (tuple, list)) or len(mu_box) != 2:
        raise ValueError("mu_box must be a (lo, hi) pair")
    a = _binary_float(mu_box[0], "mu_box lower endpoint")
    b = _binary_float(mu_box[1], "mu_box upper endpoint")
    if not 0 <= a <= b <= 1:
        raise ValueError("ridge slab requires 0 <= mu_lo <= mu_hi <= 1")
    if h is None:
        h = float(fixed.inverse_gamma.mid)
    h = _binary_float(h, "ridge scale h")
    if not h > 0:
        raise ValueError("ridge scale h must be positive")
    H = ctx.mpf(h)
    A, B = ctx.mpf(a), ctx.mpf(b)
    return RidgeSlab(
        ctx,
        fixed,
        a,
        b,
        h,
        _asinh_nonnegative(ctx, (1 - A) / H),
        _asinh_nonnegative(ctx, (1 - B) / H),
        _asinh_nonnegative(ctx, (1 + A) / H),
        _asinh_nonnegative(ctx, (1 + B) / H),
    )


def _endpoint_displacement(ctx, side, mu, u, L, h):
    if u == 0:
        return ctx.mpf(0)
    if u == 1:
        # h*sinh(asinh((1-side*mu)/h)) = 1-side*mu exactly.
        return side - mu
    return side * h * _sinh_nonnegative(ctx, u * L)


def _endpoint_eta(ctx, side, mu, u, L, h):
    if u == 1:
        return ctx.mpf(side)
    if u == 0:
        return mu
    return mu + _endpoint_displacement(ctx, side, mu, u, L, h)


def _endpoint_jacobian(ctx, u, L, h):
    return h * L * _cosh_nonnegative(ctx, u * L)


def ridge_interval_cell(ctx, slab: RidgeSlab, side: int, u_box) -> RidgeIntervalCell:
    """Enclose a correlated ``(mu,u)`` cell for one side of the eta domain.

    ``side=+1`` maps to ``[mu,1]`` and ``side=-1`` to ``[-1,mu]``; the latter
    reverses orientation, so ``jacobian`` is the positive absolute Jacobian.
    ``eta(mu,u)=mu+side*h*sinh(u*L_side(mu))`` and
    ``J=h*L_side(mu)*cosh(u*L_side(mu))``.

    On either side, ``d eta/d mu=1-u*cosh(uL)/cosh(L)>=0`` for ``0<=u<=1``.
    The displacement decreases with mu on both sides. Its u derivative has
    the sign of ``side``. ``J`` increases with L and u, while L decreases
    with mu on the plus side and increases on the minus side. These facts
    locate extrema at the corners, avoiding an independent mu/eta box hull.
    """
    if not isinstance(slab, RidgeSlab) or slab.ctx is not ctx:
        raise ValueError("ridge slab must use the supplied interval context")
    if isinstance(side, bool) or side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    if not isinstance(u_box, (tuple, list)) or len(u_box) != 2:
        raise ValueError("u_box must be a (lo, hi) pair")
    u0 = _binary_float(u_box[0], "u_box lower endpoint")
    u1 = _binary_float(u_box[1], "u_box upper endpoint")
    if not 0 <= u0 <= u1 <= 1:
        raise ValueError("u_box requires 0 <= u_lo <= u_hi <= 1")
    A, B = ctx.mpf(slab.mu_lo), ctx.mpf(slab.mu_hi)
    U0, U1 = ctx.mpf(u0), ctx.mpf(u1)
    H = ctx.mpf(slab.h)
    if side == 1:
        La, Lb = slab.L_plus_lo, slab.L_plus_hi
        delta_min = _endpoint_displacement(ctx, side, B, U0, Lb, H)
        delta_max = _endpoint_displacement(ctx, side, A, U1, La, H)
        eta_min = _endpoint_eta(ctx, side, A, U0, La, H)
        eta_max = _endpoint_eta(ctx, side, B, U1, Lb, H)
        jac_min = _endpoint_jacobian(ctx, U0, Lb, H)
        jac_max = _endpoint_jacobian(ctx, U1, La, H)
    else:
        La, Lb = slab.L_minus_lo, slab.L_minus_hi
        delta_min = _endpoint_displacement(ctx, side, B, U1, Lb, H)
        delta_max = _endpoint_displacement(ctx, side, A, U0, La, H)
        eta_min = _endpoint_eta(ctx, side, A, U1, La, H)
        eta_max = _endpoint_eta(ctx, side, B, U0, Lb, H)
        jac_min = _endpoint_jacobian(ctx, U0, La, H)
        jac_max = _endpoint_jacobian(ctx, U1, Lb, H)
    delta = ctx.mpf([delta_min.a, delta_max.b])
    eta_lo = max(ctx.mpf(-1), eta_min.a)
    eta_hi = min(ctx.mpf(1), eta_max.b)
    if eta_lo > eta_hi:
        raise ArithmeticError("mapped eta enclosure is inconsistent")
    eta = ctx.mpf([eta_lo, eta_hi])
    jacobian = ctx.mpf([max(ctx.mpf(0), jac_min.a), jac_max.b])
    mu = ctx.mpf([slab.mu_lo, slab.mu_hi])
    fixed = slab.fixed
    beta = fixed.beta
    one_minus_beta = fixed.one_minus_beta
    # Keep mu and eta correlated: eta=mu+delta. Intersect independent
    # expressions only when both are proven enclosures of the same D.
    D_correlated = one_minus_beta + beta * ((1 - mu) * (1 + mu) - mu * delta)
    D_independent = one_minus_beta + beta * (1 - mu * eta)
    D_lo = max(D_correlated.a, D_independent.a, one_minus_beta.a)
    D_hi = min(D_correlated.b, D_independent.b, (1 + beta).b)
    if D_lo > D_hi or not D_lo > 0:
        raise ArithmeticError("mapped Doppler enclosure is unusable")
    D = ctx.mpf([D_lo, D_hi])
    delta_correlated = delta + mu * one_minus_beta
    delta_independent = eta - beta * mu
    delta_lo = max(delta_correlated.a, delta_independent.a)
    delta_hi = min(delta_correlated.b, delta_independent.b)
    if delta_lo > delta_hi:
        raise ArithmeticError("mapped parallel offset enclosure is inconsistent")
    delta_ph = ctx.mpf([delta_lo, delta_hi])
    perpendicular = beta * ctx.sqrt((1 - mu) * (1 + mu))
    viewing_sine = ctx.sqrt((1 - eta) * (1 + eta))
    raw_z = perpendicular * viewing_sine / D
    z_lo = max(ctx.mpf(0), raw_z.a)
    z_hi = min(ctx.mpf(1), raw_z.b)
    # D² - beta²(1-mu²)(1-eta²) = delta_ph² + gamma^-2(1-eta²).
    defect = (delta_ph / D) ** 2 + (fixed.inverse_gamma * viewing_sine / D) ** 2
    # A lower defect endpoint always bounds z from above, even if interval
    # dependency makes the defect's upper endpoint exceed its physical
    # maximum one. Only the lower-z tightening needs defect.b <= 1.
    if defect.a > 1:
        raise ArithmeticError("mapped z defect is inconsistent")
    z_hi = min(z_hi, ctx.sqrt(1 - max(ctx.mpf(0), defect.a)).b)
    if defect.b <= 1:
        defect_z = ctx.sqrt(1 - defect)
        z_lo = max(z_lo, defect_z.a)
    # At fixed mu, differentiating beta²(1-mu²)(1-eta²)/D² shows that
    # z(eta) is maximal at eta=beta*mu. Its squared defect there is
    # gamma^-2 / (gamma^-2 + beta²(1-mu²)). This maximum decreases with
    # positive mu, so the whole slab is bounded by its mu=a endpoint.
    # The stable positive denominator avoids a spurious z_hi=1 on broad
    # ridge cells and preserves the axis limit.
    a = ctx.mpf(slab.mu_lo)
    inv_square = fixed.inverse_gamma**2
    maximum_defect = inv_square / (inv_square + beta**2 * (1 - a) * (1 + a))
    if not 0 <= maximum_defect.a <= 1:
        raise ArithmeticError("mapped maximal-z defect is inconsistent")
    z_hi = min(z_hi, ctx.sqrt(1 - maximum_defect.a).b)
    if z_lo > z_hi:
        raise ArithmeticError("mapped z enclosure is inconsistent")
    z = ctx.mpf([z_lo, z_hi])
    spacing = fixed.omega_b / (fixed.two_pi * D)
    if not spacing.a > 0:
        raise ArithmeticError("mapped harmonic spacing is unusable")
    return RidgeIntervalCell(
        (D, delta_ph, perpendicular, z, fixed.omega_b, spacing), eta, jacobian
    )


__all__ = [
    "RidgeSlab",
    "RidgeIntervalCell",
    "prepare_ridge_slab",
    "ridge_interval_cell",
]
