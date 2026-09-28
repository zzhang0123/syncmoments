"""Experimental bounded fixed-point dense integer-harmonic channels.

An interval Airy enclosure covers every integer line in an order block. The
block's interval Stokes/response/phase hull, multiplied by its integer count,
encloses the exact weighted sum without a continuum replacement. Adaptive
integer bisection tightens those hulls until a componentwise absolute
tolerance is met or a resource cap is reached. All supported lines are
included. The finite-gamma helical-orbit model remains the reference.

The calculation is conditional on mpmath's experimental interval arithmetic
and exact binary-float inputs. It is fixed-point only and internal; angular
integration, gamma/B derivatives, population tails and physical-model error
are not certified here.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from ..constants import C_SI_M
from ._certified_sparse_harmonic import (
    _component_error,
    _geometry,
    _response,
    _stokes,
)
from ._verified_airy import _airy_coefficients, _airy_on_interval, _contour_errors
from .channels import Channels
from .errors import ErrorTerm
from .high_order_harmonic import HostModes, _point
from .phase import TaylorPhase


@dataclass(frozen=True)
class DenseChannelEnclosure:
    """Fixed-point channel modes and componentwise absolute error envelopes."""

    modes: HostModes
    active_ranges: tuple[tuple[int, int], ...]
    retained_blocks: tuple[int, ...]
    evaluated_blocks: tuple[int, ...]
    absolute_error: ErrorTerm
    bessel: ErrorTerm
    block_sum: ErrorTerm
    harmonic_truncation: ErrorTerm


def _block(
    ctx,
    channels,
    channel,
    lo,
    hi,
    geometry,
    phase,
    depth_ref,
    s_depth,
    airy_coefficients,
    gaussian_gamma,
    airy_endpoint_cache,
):
    """Whole-block interval hull for every integer line in [lo,hi]."""
    mi = ctx.mpf([lo, hi])
    D, delta, perpendicular, z, omega_b, spacing = geometry
    if not (z.a >= ctx.mpf("0.9") and z.b <= 1):
        raise ValueError("dense Airy route requires 0.9 <= z <= 1")
    scale = ctx.exp(ctx.ln(2 / (mi * z)) / 3)
    # x = m(1-z)(2/(mz))^(1/3). Grouping m as m^2 removes the artificial
    # dependence between m and the decreasing scale interval. All factors
    # are nonnegative on the validated 0.9 <= z <= 1 domain.
    x = (1 - z) * ctx.exp(ctx.ln(2 * mi**2 / z) / 3)
    airy, airy_prime = _airy_on_interval(
        ctx, x, coefficients=airy_coefficients, endpoint_cache=airy_endpoint_cache
    )
    j_airy = scale * airy
    prime_airy = -(scale**2) * airy_prime
    midpoint = (lo + hi) // 2
    a = ctx.mpf((2 / (midpoint * float(z.mid))) ** (1 / 3))
    T = 10 * a
    if not T.b < ctx.pi.a:
        raise ValueError("dense Airy contour split must be below pi")
    j_error, prime_error = _contour_errors(
        ctx, mi, z, a, T, gaussian_gamma=gaussian_gamma
    )
    j_full = j_airy + ctx.mpf([-j_error, j_error])
    prime_full = prime_airy + ctx.mpf([-prime_error, prime_error])
    exact = _stokes(ctx, mi, geometry, j_full, prime_full)
    approximate = _stokes(ctx, mi, geometry, j_airy, prime_airy)
    frequency = mi * spacing
    response = _response(ctx, channels, channel, frequency)
    count = hi - lo + 1
    n_weights = 1 if phase is None else phase.n_weights
    weights = [ctx.mpc(1)]
    if phase is not None:
        tau = 2 * (ctx.mpf(C_SI_M) / frequency) ** 2
        weights[0] = ctx.exp(1j * tau * ctx.mpf(depth_ref))
        for degree in range(1, n_weights):
            weights.append(weights[-1] * (1j * tau * ctx.mpf(s_depth)) / degree)
    full = [count * response * exact[0], count * response * exact[2]]
    airy_values = [
        count * response * approximate[0],
        count * response * approximate[2],
    ]
    for weight in weights:
        full.append(count * response * exact[1] * weight)
        airy_values.append(count * response * approximate[1] * weight)
    errors = np.asarray(
        [
            _component_error(ctx, full[row], airy_values[row], complex_value=row >= 2)[
                1
            ]
            for row in range(2 + n_weights)
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(errors)):
        raise ArithmeticError("dense interval block error is not finite")
    return full, airy_values, errors


def _refine_blocks(first, last, initial, target, max_blocks, evaluate):
    """Split the largest improvable block until every error meets target."""
    initial_item = (-float(max(initial[2])), first, last, initial)
    splittable = [] if first == last else [initial_item]
    singletons = [initial_item] if first == last else []
    error_sum = initial[2].copy()
    evaluated_count = 1
    while np.any(error_sum > target):
        if not splittable:
            raise ArithmeticError("dense channel line error exceeds atol")
        if len(splittable) + len(singletons) >= max_blocks:
            raise ArithmeticError("dense channel exceeds max_blocks before atol")
        _, lo, hi, old = heapq.heappop(splittable)
        mid = (lo + hi) // 2
        left = evaluate(lo, mid)
        right = evaluate(mid + 1, hi)
        evaluated_count += 2
        error_sum = np.nextafter(error_sum - old[2] + left[2] + right[2], math.inf)
        error_sum = np.maximum(error_sum, 0)
        for child_lo, child_hi, child in ((lo, mid, left), (mid + 1, hi, right)):
            item = (-float(max(child[2])), child_lo, child_hi, child)
            if child_lo == child_hi:
                singletons.append(item)
            else:
                heapq.heappush(splittable, item)
    return singletons + splittable, evaluated_count


def certified_dense_channel_modes(
    channels: Channels,
    gamma,
    B,
    mu,
    eta,
    *,
    atol,
    phase: TaylorPhase | None = None,
    depth_ref=0.0,
    s_depth=1.0,
    max_blocks: int = 4096,
    decimal_digits: int = 70,
) -> DenseChannelEnclosure:
    """Enclose bright or faint bump channels by adaptive integer blocks.

    Requires an absolute tolerance in output units, either one scalar or one
    value per channel. The selected block hull
    includes *all* integer lines whose exact-binary-input frequencies could
    meet the open support, plus outward endpoint padding. The result is
    returned only when its whole-channel componentwise absolute error meets
    ``atol``; otherwise the function raises, including at ``max_blocks``.
    """
    gamma, B, mu, eta = _point(gamma, B, mu, eta)
    if not isinstance(channels, Channels) or channels.family != "bump":
        raise ValueError("dense enclosure requires bump Channels")
    if channels.normalisation != "unit_peak":
        raise ValueError("dense enclosure requires unit_peak normalisation")
    if phase is not None and not isinstance(phase, TaylorPhase):
        raise ValueError("dense enclosure supports TaylorPhase or no phase")
    raw_tolerances = np.asarray(atol)
    if raw_tolerances.dtype.kind in ("b", "c"):
        raise ValueError("atol must be real finite positive")
    tolerances = np.asarray(atol, dtype=float)
    if tolerances.ndim == 0:
        tolerances = np.full(channels.n_ch, float(tolerances))
    if tolerances.shape != (channels.n_ch,) or not np.all(
        np.isfinite(tolerances) & (tolerances > 0)
    ):
        raise ValueError("atol must be real finite positive per channel")
    if (
        isinstance(max_blocks, bool)
        or not isinstance(max_blocks, int)
        or max_blocks < 1
    ):
        raise ValueError("max_blocks must be an integer >= 1")
    if (
        isinstance(decimal_digits, bool)
        or not isinstance(decimal_digits, int)
        or decimal_digits < 50
    ):
        raise ValueError("decimal_digits must be an integer >= 50")
    if not math.isfinite(depth_ref) or not math.isfinite(s_depth) or s_depth <= 0:
        raise ValueError("depth_ref must be finite and s_depth finite positive")
    depth_ref, s_depth = float(depth_ref), float(s_depth)
    try:
        from mpmath.ctx_iv import MPIntervalContext  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("dense enclosures require mpmath") from exc
    ctx = MPIntervalContext()
    ctx.dps = decimal_digits
    n_weights = 1 if phase is None else phase.n_weights
    shape = (2 + n_weights, channels.n_ch)
    values = np.zeros(shape, dtype=complex)
    total_errors = np.zeros(shape, dtype=float)
    bessel_errors = np.zeros(shape, dtype=float)
    block_errors = np.zeros(shape, dtype=float)
    if B == 0 or gamma == 1 or abs(mu) == 1:
        zero = ErrorTerm(np.zeros(shape), "bound", "zero emissivity", "E_num")
        return DenseChannelEnclosure(
            HostModes(values[0].real, values[1].real, values[2:]),
            tuple((0, -1) for _ in range(channels.n_ch)),
            tuple(0 for _ in range(channels.n_ch)),
            tuple(0 for _ in range(channels.n_ch)),
            zero,
            zero,
            zero,
            zero,
        )
    geometry = _geometry(ctx, gamma, B, mu, eta)
    airy_coefficients = _airy_coefficients(ctx)
    gaussian_gamma = tuple(ctx.gamma(ctx.mpf(k + 1) / 2) for k in range(7))
    spacing = geometry[-1]
    supports = np.asarray(channels.support, dtype=float)
    ranges: list[tuple[int, int]] = []
    retained_counts: list[int] = []
    evaluated_counts: list[int] = []
    for channel, (frequency_lo, frequency_hi) in enumerate(supports):
        airy_endpoint_cache = {}
        target = tolerances[channel]
        low = ctx.mpf(float(frequency_lo)) / spacing
        high = ctx.mpf(float(frequency_hi)) / spacing
        lower_float, upper_float = float(low.a), float(high.b)
        if not (math.isfinite(lower_float) and math.isfinite(upper_float)):
            raise ValueError("active order is not representable")
        first = max(1, math.floor(math.nextafter(lower_float, -math.inf)) - 2)
        last = math.ceil(math.nextafter(upper_float, math.inf)) + 2
        if first < 1000 or last > 2**53 - 2:
            raise ValueError("dense Airy order lies outside validated range")
        ranges.append((first, last))
        initial = _block(
            ctx,
            channels,
            channel,
            first,
            last,
            geometry,
            phase,
            depth_ref,
            s_depth,
            airy_coefficients,
            gaussian_gamma,
            airy_endpoint_cache,
        )

        def evaluate(lo, hi):
            return _block(
                ctx,
                channels,
                channel,
                lo,
                hi,
                geometry,
                phase,
                depth_ref,
                s_depth,
                airy_coefficients,
                gaussian_gamma,
                airy_endpoint_cache,
            )

        blocks, evaluated_count = _refine_blocks(
            first, last, initial, target, max_blocks, evaluate
        )
        retained_counts.append(len(blocks))
        evaluated_counts.append(evaluated_count)
        full_total = [ctx.mpf(0) if row < 2 else ctx.mpc(0) for row in range(shape[0])]
        approximate_total = [
            ctx.mpf(0) if row < 2 else ctx.mpc(0) for row in range(shape[0])
        ]
        for _, _, _, (full, approximate, _) in blocks:
            for row in range(shape[0]):
                full_total[row] += full[row]
                approximate_total[row] += approximate[row]
        for row in range(shape[0]):
            centre, total, bessel, block = _component_error(
                ctx,
                full_total[row],
                approximate_total[row],
                complex_value=row >= 2,
            )
            values[row, channel] = centre
            total_errors[row, channel] = total
            bessel_errors[row, channel] = bessel
            block_errors[row, channel] = block
        if np.any(total_errors[:, channel] > target):
            raise ArithmeticError("dense channel final interval exceeds atol")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(total_errors)):
        raise ArithmeticError("dense interval output is not finite")
    return DenseChannelEnclosure(
        HostModes(values[0].real, values[1].real, values[2:]),
        tuple(ranges),
        tuple(retained_counts),
        tuple(evaluated_counts),
        ErrorTerm(
            total_errors,
            "bound",
            "whole finite-gamma integer-line channel interval for exact binary inputs; mpmath interval elementary functions",
            "E_num",
        ),
        ErrorTerm(
            bessel_errors,
            "bound",
            "uniform integer-line Airy remainder plus enclosing interval spread; conditional on mpmath interval arithmetic",
            "E_num",
        ),
        ErrorTerm(
            block_errors,
            "bound",
            "within-block weighted-summand hull; conditional on mpmath interval arithmetic",
            "E_num",
        ),
        ErrorTerm(
            np.zeros(shape),
            "bound",
            "all possibly supported integer lines included under exact binary inputs",
            "E_num",
        ),
    )


__all__ = ["DenseChannelEnclosure", "certified_dense_channel_modes"]
