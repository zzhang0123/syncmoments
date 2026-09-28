"""Finite containment checks for the ideal bump interval response."""

import math
from random import Random

import numpy as np
import pytest

from syncmoments.model import Channels
from syncmoments.model._certified_sparse_harmonic import _response


def _fixture():
    MPIntervalContext = pytest.importorskip("mpmath.ctx_iv").MPIntervalContext
    ctx = MPIntervalContext()
    ctx.dps = 70
    channels = Channels.bump([1e8], [2e7])
    lo, hi = (float(v) for v in np.asarray(channels.support[0]))
    centre = float(channels.centres_hz[0])
    width = float(channels.widths_hz[0])
    return ctx, channels, lo, hi, centre, width


def _ideal_point(ctx, channels, frequency):
    lo, hi = (ctx.mpf(float(v)) for v in np.asarray(channels.support[0]))
    nu = ctx.mpf(float(frequency))
    if nu <= lo or nu >= hi:
        return ctx.mpf(0)
    centre = ctx.mpf(float(channels.centres_hz[0]))
    width = ctx.mpf(float(channels.widths_hz[0]))
    q = ((nu - centre) / width) ** 2
    if q.a >= 1:
        return ctx.mpf(0)
    return ctx.exp(1 - 1 / (1 - q))


def _contains(interval, value):
    assert interval.a <= value.a
    assert interval.b >= value.b


def test_proved_zero_outside_both_support_edges():
    ctx, channels, lo, hi, _, width = _fixture()
    for a, b in [
        (lo - width, lo),
        (hi, hi + width),
        (lo, lo),
        (hi, hi),
    ]:
        assert _response(ctx, channels, 0, ctx.mpf([a, b])) == 0


def test_crossing_either_edge_has_zero_lower_and_sharp_upper():
    ctx, channels, lo, hi, _, width = _fixture()
    for a, b in [
        (lo - width / 4, lo + width / 4),
        (hi - width / 4, hi + width / 4),
    ]:
        hull = _response(ctx, channels, 0, ctx.mpf([a, b]))
        assert hull.a == 0
        assert 0 < hull.b < 1
        for frequency in (a, a + (b - a) / 4, (a + b) / 2, b):
            _contains(hull, _ideal_point(ctx, channels, frequency))


def test_center_crossing_and_interior_wing_use_nonzero_lower():
    ctx, channels, _, _, centre, width = _fixture()
    central = _response(
        ctx, channels, 0, ctx.mpf([centre - 0.2 * width, centre + 0.3 * width])
    )
    assert 0 < central.a < 1
    assert central.b == 1
    wing = _response(
        ctx, channels, 0, ctx.mpf([centre + 0.4 * width, centre + 0.6 * width])
    )
    assert 0 < wing.a < wing.b < 1
    for frequency in (centre, centre - 0.1 * width, centre + 0.3 * width):
        _contains(central, _ideal_point(ctx, channels, frequency))
    for frequency in (centre + 0.45 * width, centre + 0.55 * width):
        _contains(wing, _ideal_point(ctx, channels, frequency))


def test_binary_perturbations_around_edges_do_not_create_positive_lower():
    ctx, channels, lo, hi, _, _ = _fixture()
    for edge in (lo, hi):
        a = math.nextafter(edge, -math.inf)
        b = math.nextafter(edge, math.inf)
        hull = _response(ctx, channels, 0, ctx.mpf([a, b]))
        assert hull.a == 0
        assert hull.b <= 1
        for frequency in (a, edge, b):
            _contains(hull, _ideal_point(ctx, channels, frequency))


def test_center_touching_and_binary_adjacent_intervals_keep_unit_peak():
    ctx, channels, _, _, centre, _ = _fixture()
    left = math.nextafter(centre, -math.inf)
    right = math.nextafter(centre, math.inf)
    for a, b in [(left, centre), (centre, right), (left, right)]:
        hull = _response(ctx, channels, 0, ctx.mpf([a, b]))
        assert hull.b == 1
        _contains(hull, _ideal_point(ctx, channels, centre))
    for frequency in (left, right):
        hull = _response(ctx, channels, 0, ctx.mpf(frequency))
        _contains(hull, _ideal_point(ctx, channels, frequency))


def test_random_scalar_samples_are_contained_across_edges_center_and_wings():
    ctx, channels, _, _, centre, width = _fixture()
    rng = Random(141)
    for _ in range(100):
        a = centre + rng.uniform(-1.5, 1.2) * width
        b = a + rng.uniform(0.005, 0.7) * width
        hull = _response(ctx, channels, 0, ctx.mpf([a, b]))
        assert hull.a >= 0 and hull.b <= 1
        for fraction in (0.1, 0.5, 0.9):
            frequency = a + fraction * (b - a)
            _contains(hull, _ideal_point(ctx, channels, frequency))
