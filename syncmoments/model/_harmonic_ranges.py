"""Static harmonic intervals for channel support pruning.

The interval calculation is eager: it uses exact rational arithmetic on the
stored binary floating-point inputs and constants.  Its zero-contribution
claim is conditional on the declared gamma/B support and on the mathematical
line-frequency formula.  Floating-point evaluation error remains a separate
numerical limitation of the harmonic kernel.
"""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from ..constants import C_CGS, E_ESU, M_E


def normalise_intervals(intervals, m_max: int) -> tuple[tuple[int, int], ...]:
    """Validate and merge inclusive integer intervals inside ``1..m_max``."""
    if not isinstance(intervals, (tuple, list)):
        raise ValueError("mode_intervals must be a sequence of (lo, hi) pairs")
    parsed = []
    for pair in intervals:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("mode_intervals entries must be (lo, hi) pairs")
        lo, hi = pair
        if any(isinstance(x, bool) or not isinstance(x, int) for x in pair):
            raise ValueError("mode_intervals endpoints must be integers")
        if not 1 <= lo <= hi <= m_max:
            raise ValueError(f"mode_intervals must lie inside 1..m_max={m_max}")
        parsed.append((lo, hi))
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(parsed):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def intersect_intervals(left, right) -> tuple[tuple[int, int], ...]:
    """Intersection of sorted, disjoint inclusive integer intervals."""
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        lo = max(left[i][0], right[j][0])
        hi = min(left[i][1], right[j][1])
        if lo <= hi:
            out.append((lo, hi))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return tuple(out)


def subtract_intervals(left, right) -> tuple[tuple[int, int], ...]:
    """Intervals in ``left`` missing from ``right`` (both sorted and disjoint)."""
    out = []
    for lo, hi in left:
        cursor = lo
        for r_lo, r_hi in right:
            if r_hi < cursor:
                continue
            if r_lo > hi:
                break
            if cursor < r_lo:
                out.append((cursor, min(hi, r_lo - 1)))
            cursor = max(cursor, r_hi + 1)
            if cursor > hi:
                break
        if cursor <= hi:
            out.append((cursor, hi))
    return tuple(out)


def _float_fraction(value, name: str) -> Fraction:
    try:
        scalar = np.asarray(value, dtype=float)
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError
        return Fraction.from_float(float(scalar))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite concrete scalar") from exc


def support_intervals(support, channels, m_max: int, reference=None, *, separate=False):
    """A conservative union of orders that can meet a channel support.

    For ``gamma <= g``, ``B_lo <= B <= B_hi`` and ``mu*eta in [-1,1]``,
    ``nu_m/m`` lies in ``[K B_lo/(g(1+beta)), K B_hi g(1+beta)]``.  We use
    ``beta <= 1-1/(2g^2)`` (from ``sqrt(1-x) <= 1-x/2``), and exact rational
    arithmetic on the stored float inputs to round both endpoints outward.
    The extra one order at each end also makes edge coincidences harmless.
    """
    g = _float_fraction(support.gamma[1], "support.gamma[1]")
    B_lo = _float_fraction(support.B[0], "support.B[0]")
    B_hi = _float_fraction(support.B[1], "support.B[1]")
    if reference is not None:
        # The Taylor basis is evaluated at the reference even when that point
        # lies outside the declared population support.
        g = max(g, _float_fraction(reference.gamma0, "reference.gamma0"))
        B_ref = _float_fraction(reference.B0, "reference.B0")
        B_lo, B_hi = min(B_lo, B_ref), max(B_hi, B_ref)
    if g <= 1 or B_lo <= 0 or B_hi < B_lo:
        raise ValueError("support needs gamma_max > 1 and 0 < B_min <= B_max")
    edges = np.asarray(channels.support, dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.all(np.isfinite(edges)):
        raise ValueError("channels.support must be a finite (n_ch, 2) array")
    if np.any(edges[:, 0] <= 0) or np.any(edges[:, 1] <= edges[:, 0]):
        raise ValueError("channel support needs 0 < lo < hi")
    beta_upper = 1 - Fraction(1, 2) / (g * g)
    K = Fraction.from_float(E_ESU) / (
        2
        * Fraction.from_float(math.pi)
        * Fraction.from_float(M_E)
        * Fraction.from_float(C_CGS)
    )
    k_min_lower = K * B_lo / (g * (1 + beta_upper))
    k_max_upper = K * B_hi * g * (1 + beta_upper)
    intervals = []
    for nu_lo, nu_hi in edges:
        lo = max(1, math.floor(Fraction.from_float(float(nu_lo)) / k_max_upper) - 1)
        hi = min(m_max, math.ceil(Fraction.from_float(float(nu_hi)) / k_min_lower) + 1)
        intervals.append(((lo, hi),) if lo <= hi else ())
    if separate:
        return tuple(intervals)
    return normalise_intervals(
        [interval for channel in intervals for interval in channel], m_max
    )


__all__ = [
    "normalise_intervals",
    "intersect_intervals",
    "subtract_intervals",
    "support_intervals",
]
