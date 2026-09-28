"""Independent finite checks of the integer-order interval Bessel route."""

import pytest

from syncmoments.model._verified_bessel import integer_bessel_enclosure


@pytest.mark.parametrize("m,z", [(100, 0.9), (1000, 0.999), (1000, 1.0)])
def test_shifted_contour_intervals_contain_high_precision_bessel(m, z):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 80
    result = integer_bessel_enclosure(m, z, abs_tolerance=1e-8)
    argument = m * mp.mpf(z)
    actual = mp.besselj(m, argument)
    actual_prime = (mp.besselj(m - 1, argument) - mp.besselj(m + 1, argument)) / 2
    assert float(result.j.a) <= actual <= float(result.j.b)
    assert float(result.j_prime.a) <= actual_prime <= float(result.j_prime.b)
    assert float(result.truncation_error) <= 1e-8
    assert float(result.derivative_truncation_error) <= 1e-8
    assert result.nodes < 2000


def test_zero_argument_and_resource_cap():
    pytest.importorskip("mpmath")
    first = integer_bessel_enclosure(1, 0.0)
    second = integer_bessel_enclosure(2, 0.0)
    assert first.nodes == second.nodes == 0
    assert float(first.j_prime.a) == 0.5
    assert float(second.j_prime.a) == 0.0
    with pytest.raises(ValueError, match="exceeds max_nodes"):
        integer_bessel_enclosure(1000, 1.0, abs_tolerance=1e-12, max_nodes=8)
    with pytest.raises(ValueError, match="positive integer"):
        integer_bessel_enclosure(0, 1.0)
    with pytest.raises(ValueError, match="z >= 0.5"):
        integer_bessel_enclosure(100, 0.25)


def test_input_interval_covers_both_bessel_endpoints():
    mp = pytest.importorskip("mpmath")
    from mpmath.ctx_iv import MPIntervalContext

    ctx = MPIntervalContext()
    ctx.dps = 70
    z = ctx.mpf(["0.9", "0.9000000000001"])
    result = integer_bessel_enclosure(100, z, abs_tolerance=1e-8)
    mp.mp.dps = 70
    for endpoint in (z.a, z.b):
        argument = 100 * mp.mpf(float(endpoint))
        actual = mp.besselj(100, argument)
        assert float(result.j.a) <= actual <= float(result.j.b)


def test_odd_node_conjugate_pairing_contains_high_precision_values():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 80
    result = integer_bessel_enclosure(100, 0.9, abs_tolerance=8e-9)
    assert result.nodes % 2 == 1
    argument = 100 * mp.mpf(0.9)
    actual = mp.besselj(100, argument)
    actual_prime = (mp.besselj(99, argument) - mp.besselj(101, argument)) / 2
    assert float(result.j.a) <= actual <= float(result.j.b)
    assert float(result.j_prime.a) <= actual_prime <= float(result.j_prime.b)
