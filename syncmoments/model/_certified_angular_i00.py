"""Conditional whole-angle I00 certificate for one fixed-particle bump.

I00 = 0.5 integral_{mu=0}^1 integral_{eta=-1}^1 I(mu,eta) d eta d mu.
The factor follows positive-pitch parity and the (0,0) Legendre normalisation
in build_high_order_basis. Two correlated ridge patches cover the entire eta
interval. The proof assumes exact binary inputs, the ideal stored-support
unit-peak bump, and mpmath's experimental interval elementary functions.
The optional derivative extension also accumulates coarse whole-angle
gamma/B magnitudes at the same fixed reference. Neither route certifies
populations, physical response-model fidelity, or a parameter displacement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
from numbers import Real
from time import perf_counter

import numpy as np

from ._angular_faint_bounds import (
    angular_faint_magnitude_bounds,
    prepare_faint_bound_constants,
    prepare_faint_cell,
)
from ._angular_global_bounds import (
    angular_global_first_derivative_bounds,
    angular_global_magnitude_bounds,
    prepare_global_bound_constants,
    prepare_global_cell,
)
from ._certified_dense_harmonic import _block
from ._interval_angular_geometry import (
    active_channel_ranges,
    prepare_angular_geometry,
    prepare_channel_supports,
)
from ._ridge_interval_cells import prepare_ridge_slab, ridge_interval_cell
from ._verified_airy import _airy_coefficients
from .channels import Channels
from .phase import TaylorPhase


@dataclass(frozen=True)
class AngularI00Certificate:
    """MP enclosure and error; optional derivative magnitudes use the same cells."""

    interval: object
    absolute_error: float
    cells: int
    evaluated_cells: int
    splits: int
    route_counts: tuple[tuple[str, int], ...]
    mode_blocks_evaluated: int
    mode_blocks_retained: int
    elapsed_seconds: float
    derivative_upper: tuple[tuple[object, ...], tuple[object, ...]] | None = None


class AngularI00Limit(ArithmeticError):
    """A resource cap prevented the requested certificate."""


@dataclass(frozen=True)
class _Leaf:
    slab: object
    side: int
    u_lo: float
    u_hi: float
    contribution: object
    route: str
    z: object
    active_range: tuple[int, int]
    derivative_contribution: tuple[tuple[object, ...], tuple[object, ...]] | None


def _binary_float(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite binary float")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite binary float") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(
            f"{name} must be a finite {'positive ' if positive else ''}binary float"
        )
    return result


def _cap(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _intersect(ctx, x, y):
    lo, hi = max(x.a, y.a), min(x.b, y.b)
    if lo > hi:
        raise ArithmeticError("independent angular enclosures disagree")
    return ctx.mpf([lo, hi])


def _outward_error(value):
    if not value.b >= 0:
        raise ArithmeticError("angular error is negative")
    result = float(value.b)
    if not math.isfinite(result):
        raise ArithmeticError("angular error is not representable")
    if value.b == 0:
        return 0.0
    outward = math.nextafter(result, math.inf)
    if not math.isfinite(outward):
        raise ArithmeticError("angular error cannot round outward")
    return outward


def certify_angular_i00(
    channels: Channels,
    gamma,
    B,
    fast_i00,
    *,
    atol,
    phase=None,
    first_derivatives: bool = False,
    depth_ref=0.0,
    s_depth=1.0,
    max_cells: int = 512,
    max_evaluated_cells: int = 1024,
    max_mode_blocks: int = 64,
    max_blocks_per_cell: int = 4,
    decimal_digits: int = 70,
) -> AngularI00Certificate:
    """Enclose exact I00 and compare it with a supplied fast binary64 value.

    Success requires the outward maximum distance from fast_i00 to both
    interval endpoints to be at most atol. A cap raises AngularI00Limit
    with work counters. The returned MP interval must be rounded outward
    separately if its endpoints are published as binary64. The optional
    ``first_derivatives`` branch accumulates coarse derivative magnitudes
    on those angular cells; it does not certify a parameter-domain remainder.
    """
    if not isinstance(channels, Channels) or channels.n_ch != 1:
        raise ValueError("angular I00 requires one Channels channel")
    if channels.family != "bump" or channels.normalisation != "unit_peak":
        raise ValueError("angular I00 requires a unit-peak bump")
    if not isinstance(first_derivatives, bool):
        raise ValueError("first_derivatives must be a bool")
    if first_derivatives:
        if not isinstance(phase, TaylorPhase):
            raise ValueError("derivative angular certificate requires TaylorPhase")
        depth_ref = _binary_float(depth_ref, "depth_ref")
        s_depth = _binary_float(s_depth, "s_depth", positive=True)
    elif phase is not None:
        raise ValueError("angular I00 prototype supports no phase")
    gamma = _binary_float(gamma, "gamma", positive=True)
    B = _binary_float(B, "B", positive=True)
    if gamma <= 1:
        raise ValueError("gamma must exceed one")
    fast_i00 = _binary_float(fast_i00, "fast_i00")
    atol = _binary_float(atol, "atol", positive=True)
    max_cells = _cap(max_cells, "max_cells", 2)
    max_evaluated_cells = _cap(max_evaluated_cells, "max_evaluated_cells", 2)
    max_mode_blocks = _cap(max_mode_blocks, "max_mode_blocks", 0)
    max_blocks_per_cell = _cap(max_blocks_per_cell, "max_blocks_per_cell", 1)
    decimal_digits = _cap(decimal_digits, "decimal_digits", 50)
    from mpmath.ctx_iv import MPIntervalContext  # type: ignore[import-untyped]

    started = perf_counter()
    ctx = MPIntervalContext()
    ctx.dps = decimal_digits
    fixed = prepare_angular_geometry(ctx, gamma, B)
    supports = prepare_channel_supports(ctx, np.asarray(channels.support))
    frequency_lo = float(np.asarray(channels.support)[0, 0])
    frequency_hi = float(np.asarray(channels.support)[0, 1])
    channel_width = float(np.asarray(channels.widths_hz)[0])
    if first_derivatives:
        # The stored open support must contain both true bump zeros when its
        # binary inputs are interpreted exactly. Otherwise a clipped bump
        # jumps at a stored edge, and moving-line differentiation would need
        # an extra boundary term.
        centre = ctx.mpf(float(np.asarray(channels.centres_hz)[0]))
        width = ctx.mpf(channel_width)
        support_lo, support_hi = supports.endpoints[0]
        if support_lo.b > (centre - width).a or support_hi.a < (centre + width).b:
            raise ValueError(
                "derivative certificate requires bump support to contain its exact zeros"
            )
    global_constants = prepare_global_bound_constants(ctx)
    faint_constants = prepare_faint_bound_constants(ctx)
    routes: Counter[str] = Counter()
    evaluated_cells = splits = mode_blocks_evaluated = mode_blocks_retained = 0
    airy_coefficients = gaussian_gamma = None
    airy_endpoint_cache = {}
    leaves: dict[int, _Leaf] = {}

    def evaluate(slab, side, u_lo, u_hi):
        nonlocal evaluated_cells, mode_blocks_evaluated, mode_blocks_retained
        nonlocal airy_coefficients, gaussian_gamma
        if evaluated_cells >= max_evaluated_cells:
            raise AngularI00Limit(
                f"max_evaluated_cells={max_evaluated_cells}; "
                f"cells={len(leaves)}, splits={splits}, "
                f"mode_blocks={mode_blocks_evaluated}"
            )
        evaluated_cells += 1
        mapped = ridge_interval_cell(ctx, slab, side, (u_lo, u_hi))
        geometry = mapped.geometry
        first, last = active_channel_ranges(ctx, supports, geometry[-1])[0]
        area = (
            ctx.mpf("0.5")
            * (ctx.mpf(slab.mu_hi) - ctx.mpf(slab.mu_lo))
            * (ctx.mpf(u_hi) - ctx.mpf(u_lo))
            * mapped.jacobian
        )
        if first == 0:
            intensity = ctx.mpf(0)
            route = "zero"
        else:
            prepared = prepare_global_cell(
                ctx,
                geometry,
                constants=global_constants,
                fixed=fixed,
                mu_abs_lower=slab.mu_lo,
            )
            intensity = angular_global_magnitude_bounds(
                ctx, prepared, first, last, frequency_lo
            )[0]
            route = "global"
            z = geometry[3]
            if first >= 3 and z.b < 1:
                faint_cell = prepare_faint_cell(
                    ctx,
                    geometry,
                    constants=faint_constants,
                    fixed=fixed,
                    mu_abs_lower=slab.mu_lo,
                )
                if faint_cell.z_hi == 0 or first * faint_cell.root_lower > 1:
                    faint = angular_faint_magnitude_bounds(
                        ctx, faint_cell, first, last, frequency_lo
                    )[0]
                    if faint.b < intensity.b:
                        route = "faint"
                    intensity = _intersect(ctx, intensity, faint)

            eligible = (
                slab.mu_hi - slab.mu_lo <= 0.125
                and u_hi - u_lo <= 0.125
                and z.a >= ctx.mpf("0.9")
                and z.b <= 1
                and first >= 1000
                and mode_blocks_evaluated < max_mode_blocks
                # This only schedules costly Airy work. The all-z/faint
                # enclosure still covers every skipped cell and mode.
                and (area * intensity).b > 10 * ctx.mpf(atol) / max_cells
            )
            if eligible:
                orders = ctx.mpf([first, last])
                x = (1 - z) * ctx.exp(ctx.ln(2 * orders**2 / z) / 3)
                eligible = x.a >= 0 and x.b <= 10
            if eligible:
                if airy_coefficients is None:
                    airy_coefficients = _airy_coefficients(ctx)
                    gaussian_gamma = tuple(
                        ctx.gamma(ctx.mpf(k + 1) / 2) for k in range(7)
                    )

                def block(lo, hi):
                    nonlocal mode_blocks_evaluated
                    mode_blocks_evaluated += 1
                    full, _, _ = _block(
                        ctx,
                        channels,
                        0,
                        lo,
                        hi,
                        geometry,
                        None,
                        0.0,
                        1.0,
                        airy_coefficients,
                        gaussian_gamma,
                        airy_endpoint_cache,
                    )
                    return full[0]

                try:
                    pieces = [(first, last, block(first, last))]
                except (ValueError, ArithmeticError):
                    # The proved whole-range envelope remains available.
                    pieces = []
                if pieces:
                    block_sum = pieces[0][2]
                    while (
                        len(pieces) < max_blocks_per_cell
                        and mode_blocks_evaluated + 2 <= max_mode_blocks
                    ):
                        candidate = max(
                            (i for i, (lo, hi, _) in enumerate(pieces) if lo < hi),
                            key=lambda i: float((pieces[i][2].b - pieces[i][2].a).b),
                            default=None,
                        )
                        if candidate is None:
                            break
                        lo, hi, _ = pieces[candidate]
                        middle = (lo + hi) // 2
                        try:
                            left = block(lo, middle)
                            right = block(middle + 1, hi)
                        except (ValueError, ArithmeticError):
                            break
                        pieces[candidate : candidate + 1] = [
                            (lo, middle, left),
                            (middle + 1, hi, right),
                        ]
                        refined = sum((piece[2] for piece in pieces), ctx.mpf(0))
                        block_sum = _intersect(ctx, block_sum, refined)
                    intensity = _intersect(ctx, intensity, block_sum)
                    mode_blocks_retained += len(pieces)
                    route = "dense"
        derivatives = None
        if first_derivatives:
            eta_abs_upper = max(abs(mapped.eta.a), abs(mapped.eta.b))
            gamma_bounds, B_bounds = angular_global_first_derivative_bounds(
                ctx,
                geometry,
                fixed,
                first,
                last,
                frequency_lo,
                frequency_hi,
                channel_width,
                B,
                slab.mu_hi,
                eta_abs_upper,
                depth_ref=depth_ref,
                s_depth=s_depth,
                max_b=phase.max_b(),
                constants=global_constants,
            )
            derivatives = (
                tuple(area * bound for bound in gamma_bounds),
                tuple(area * bound for bound in B_bounds),
            )
        contribution = area * intensity
        if not contribution.a >= 0 or not math.isfinite(float(contribution.b)):
            raise ArithmeticError("angular cell contribution is not representable")
        routes[route] += 1
        return _Leaf(
            slab,
            side,
            u_lo,
            u_hi,
            contribution,
            route,
            geometry[3],
            (first, last),
            derivatives,
        )

    def width(leaf):
        return float((leaf.contribution.b - leaf.contribution.a).b)

    def dominant_leaf():
        leaf = max(leaves.values(), key=width)
        return (
            f"top_mu=({leaf.slab.mu_lo:.17g},{leaf.slab.mu_hi:.17g}),"
            f"top_u=({leaf.u_lo:.17g},{leaf.u_hi:.17g}),"
            f"side={leaf.side},route={leaf.route},"
            f"z=({float(leaf.z.a):.6g},{float(leaf.z.b):.6g}),"
            f"active={leaf.active_range},"
            f"projected_U={float(leaf.contribution.b):.6g},"
            f"projected_width={width(leaf):.6g}"
        )

    queue: list[tuple[float, int]] = []
    serial = 0
    slab = prepare_ridge_slab(ctx, fixed, (0.0, 1.0))
    for side in (-1, 1):
        leaf = evaluate(slab, side, 0.0, 1.0)
        leaves[serial] = leaf
        heapq.heappush(queue, (-width(leaf), serial))
        serial += 1
    previous_total = None
    fast_interval = ctx.mpf(fast_i00)
    while True:
        total = sum((leaf.contribution for leaf in leaves.values()), ctx.mpf(0))
        if previous_total is not None:
            total = _intersect(ctx, total, previous_total)
        previous_total = total
        error = max(abs(fast_interval - total.a).b, abs(total.b - fast_interval).b)
        error_float = _outward_error(error)
        if error_float <= atol:
            derivative_upper = None
            if first_derivatives:
                derivative_upper = tuple(
                    tuple(
                        sum(
                            (
                                leaf.derivative_contribution[direction][row]
                                for leaf in leaves.values()
                            ),
                            ctx.mpf(0),
                        )
                        for row in range(phase.max_b() + 2)
                    )
                    for direction in range(2)
                )
            return AngularI00Certificate(
                total,
                error_float,
                len(leaves),
                evaluated_cells,
                splits,
                tuple(sorted(routes.items())),
                mode_blocks_evaluated,
                mode_blocks_retained,
                perf_counter() - started,
                derivative_upper,
            )
        if len(leaves) >= max_cells or evaluated_cells + 2 > max_evaluated_cells:
            raise AngularI00Limit(
                f"cell cap; error={error_float:.6g}, atol={atol:.6g}, "
                f"cells={len(leaves)}, evaluated_cells={evaluated_cells}, "
                f"splits={splits}, routes={dict(routes)}, "
                f"mode_blocks={mode_blocks_evaluated}, "
                f"retained_blocks={mode_blocks_retained}, {dominant_leaf()}"
            )
        _, key = heapq.heappop(queue)
        old = leaves.pop(key)
        mu_width = old.slab.mu_hi - old.slab.mu_lo
        u_width = old.u_hi - old.u_lo
        # An axis-touching slab can combine p at its lower-mu edge with
        # D at mu=1. Resolve that physical layer down to a few 1-beta
        # widths before spending splits on the mapped viewing coordinate.
        axis_layer = old.slab.mu_hi == 1.0 and mu_width > 4 * float(
            fixed.one_minus_beta.b
        )
        if axis_layer or mu_width >= u_width:
            middle = (old.slab.mu_lo + old.slab.mu_hi) / 2
            if not old.slab.mu_lo < middle < old.slab.mu_hi:
                raise AngularI00Limit("mu split reached binary64 resolution")
            pieces = (
                (
                    prepare_ridge_slab(
                        ctx, fixed, (old.slab.mu_lo, middle), h=old.slab.h
                    ),
                    old.side,
                    old.u_lo,
                    old.u_hi,
                ),
                (
                    prepare_ridge_slab(
                        ctx, fixed, (middle, old.slab.mu_hi), h=old.slab.h
                    ),
                    old.side,
                    old.u_lo,
                    old.u_hi,
                ),
            )
        else:
            middle = (old.u_lo + old.u_hi) / 2
            if not old.u_lo < middle < old.u_hi:
                raise AngularI00Limit("u split reached binary64 resolution")
            pieces = (
                (old.slab, old.side, old.u_lo, middle),
                (old.slab, old.side, middle, old.u_hi),
            )
        for piece in pieces:
            leaf = evaluate(*piece)
            leaves[serial] = leaf
            heapq.heappush(queue, (-width(leaf), serial))
            serial += 1
        splits += 1


__all__ = ["AngularI00Certificate", "AngularI00Limit", "certify_angular_i00"]
