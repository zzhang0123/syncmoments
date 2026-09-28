"""Finite checks of the restricted Airy integer-line enclosure."""

import pytest

from syncmoments.model._verified_airy import (
    _airy_coefficients,
    _airy_on_interval,
    _airy_series,
    integer_bessel_airy_enclosure,
)
from syncmoments.model._verified_bessel import integer_bessel_enclosure


@pytest.mark.parametrize("x", [0.0, 0.3, 1.0, 5.0, 10.0])
def test_adaptive_airy_series_contains_high_precision_values(x):
    mp = pytest.importorskip("mpmath")
    from mpmath.ctx_iv import MPIntervalContext

    ctx = MPIntervalContext()
    ctx.dps = 70
    coefficients = _airy_coefficients(ctx)
    airy, prime, _, _ = _airy_series(ctx, ctx.mpf(x), coefficients=coefficients)
    with mp.workdps(100):
        point = mp.mpf(x)
        reference = mp.airyai(point)
        reference_prime = mp.airyai(point, derivative=1)
        assert airy.a <= reference <= airy.b
        assert prime.a <= reference_prime <= prime.b


def test_reused_airy_endpoint_cache_matches_fresh_interval_evaluations():
    pytest.importorskip("mpmath")
    from mpmath.ctx_iv import MPIntervalContext

    ctx = MPIntervalContext()
    ctx.dps = 70
    coefficients = _airy_coefficients(ctx)
    cache = {}
    intervals = [ctx.mpf(["0.2", "0.6"]), ctx.mpf(["0.2", "0.8"])]
    for argument in intervals:
        cached = _airy_on_interval(
            ctx, argument, coefficients=coefficients, endpoint_cache=cache
        )
        fresh = _airy_on_interval(ctx, argument, coefficients=coefficients)
        for with_cache, without_cache in zip(cached, fresh):
            assert with_cache.a == without_cache.a
            assert with_cache.b == without_cache.b
    assert len(cache) == 3


def test_airy_enclosure_contains_high_precision_integer_line():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 80
    m, z = 1000, 0.999
    result = integer_bessel_airy_enclosure(m, z)
    argument = m * mp.mpf(z)
    exact = mp.besselj(m, argument)
    exact_prime = (mp.besselj(m - 1, argument) - mp.besselj(m + 1, argument)) / 2
    assert result.j.a <= exact <= result.j.b
    assert result.j_prime.a <= exact_prime <= result.j_prime.b
    assert float(result.j_error) < 1e-3


def test_large_order_airy_and_contour_intervals_overlap():
    pytest.importorskip("mpmath")
    m, z = 1_000_000, 0.999999
    airy = integer_bessel_airy_enclosure(m, z)
    contour = integer_bessel_enclosure(m, z, abs_tolerance=1e-9)
    assert airy.j.a <= contour.j.b and contour.j.a <= airy.j.b
    assert airy.j_prime.a <= contour.j_prime.b
    assert contour.j_prime.a <= airy.j_prime.b
    assert float(airy.j_error) < 1e-6
    assert float(airy.j_prime_error) < 1e-7


def test_airy_route_fails_closed_outside_turning_regime():
    pytest.importorskip("mpmath")
    with pytest.raises(ValueError, match="m must be"):
        integer_bessel_airy_enclosure(100, 1.0)
    with pytest.raises(ValueError, match="0.9 <= z"):
        integer_bessel_airy_enclosure(1000, 0.8)
    with pytest.raises(ValueError, match="0 <= x <= 10"):
        integer_bessel_airy_enclosure(1000, 0.9)
