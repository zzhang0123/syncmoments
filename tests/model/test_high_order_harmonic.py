"""Finite checks of the provisional host-only high-harmonic value backend.

These overlap tests do not bound the extrapolation to Galactic harmonic orders.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncmoments.constants import C_CGS, E_ESU, M_E
from syncmoments.model import (
    Channels,
    HarmonicKernel,
    TaylorPhase,
    high_order_channel_modes,
    high_order_line_powers,
)
from syncmoments.model._kernel_helpers import phase_coordinate, phase_weights
from syncmoments.model.high_order_harmonic import (
    _host_bump_response_and_slope,
    _host_phase_weights,
    _host_response,
    _line_powers_with_derivatives,
    _prepare_host_responses,
    high_order_channel_derivatives,
)


def _nu_b(gamma, B=1.0):
    return E_ESU * B / (2 * np.pi * gamma * M_E * C_CGS)


@pytest.mark.parametrize("normalisation", ["unit_peak", "unit_integral"])
def test_host_bump_response_matches_jax_at_edges_and_interior(normalisation):
    channels = Channels.bump([1e8], [2e7], normalisation=normalisation)
    frequency = np.array([7.9e7, 8e7, 8.1e7, 9e7, 1e8, 1.1e8, 1.19e8, 1.2e8])
    expected = np.asarray(channels(frequency))[0]
    actual = _host_response(channels, 0)(frequency)
    assert_allclose(actual, expected, rtol=5e-15, atol=0)
    assert actual[0] == actual[1] == actual[-1] == 0


def test_host_taylor_phase_matches_jax_weights():
    frequency = np.array([8e7, 1e8, 1.2e8])
    phase = TaylorPhase(2)
    expected = np.asarray(phase_weights(phase, phase_coordinate(frequency), 0.1, 0.01))
    actual = _host_phase_weights(phase, frequency, 0.1, 0.01)
    assert_allclose(actual, expected, rtol=5e-15, atol=0)


def test_prepared_response_rejects_different_channel_instance():
    first = Channels.bump([1e8], [2e7])
    second = Channels.bump([1e8], [2e7])
    for derivatives, evaluator in (
        (False, high_order_channel_modes),
        (True, high_order_channel_derivatives),
    ):
        prepared = _prepare_host_responses(first, derivatives=derivatives)
        with pytest.raises(ValueError, match="do not match"):
            evaluator(
                second,
                20.0,
                1.0,
                0.3,
                -0.2,
                _prepared_responses=prepared,
            )


def test_derivative_refuses_floating_clipped_bump_edges():
    # Restore the unpadded rounded support to emulate a stale/custom channel;
    # its upper edge lies below exact binary c+w and clips the smooth profile.
    nominal = Channels.bump([1e8], [5e-8])
    rounded = jnp.stack(
        [
            nominal.centres_hz - nominal.widths_hz,
            nominal.centres_hz + nominal.widths_hz,
        ],
        axis=-1,
    )
    clipped = eqx.tree_at(lambda c: c.support, nominal, rounded)
    with pytest.raises(ValueError, match="exact smooth zeros"):
        high_order_channel_derivatives(clipped, 20.0, 1.0, 0.3, -0.2)


def test_low_orders_match_existing_harmonic_kernel_with_line_phase():
    nu_b = _nu_b(20.0)
    channels = Channels.bump([5.0 * nu_b], [2.0 * nu_b])
    phase = TaylorPhase(1)
    kwargs = dict(phase=phase, depth_ref=2e-6, s_depth=1e-7)
    old = HarmonicKernel(40).channel_modes(channels, 20.0, 1.0, 0.3, -0.6, **kwargs)
    new = high_order_channel_modes(channels, 20.0, 1.0, 0.3, -0.6, **kwargs)
    assert new.routes == ("direct",)
    assert new.quadrature.kind == "not_applicable"
    assert new.harmonic_sum.kind == "not_applicable"
    assert new.bessel.kind == new.floating_point.kind == "unbounded"
    assert_allclose(new.modes.I, old.I, rtol=2e-12)
    assert_allclose(new.modes.V, old.V, rtol=2e-12)
    assert_allclose(new.modes.P, old.P, rtol=2e-12)


def test_dense_smooth_overlap_matches_explicit_integer_sum():
    nu_b = _nu_b(100.0)
    channels = Channels.bump([3000.0 * nu_b], [1000.0 * nu_b])
    kwargs = dict(phase=TaylorPhase(1), depth_ref=0.01, s_depth=0.001)
    direct = high_order_channel_modes(
        channels, 100.0, 1.0, 0.3, 0.2, direct_limit=10000, **kwargs
    )
    dense = high_order_channel_modes(
        channels, 100.0, 1.0, 0.3, 0.2, direct_limit=64, **kwargs
    )
    assert direct.routes == ("direct",)
    assert dense.routes == ("corrected_integral",)
    assert dense.harmonic_sum.kind == "unbounded"
    assert dense.quadrature.kind == "estimate"
    assert dense.quadrature_converged == (True,)
    scale = float(direct.modes.I[0])
    assert_allclose(dense.modes.I, direct.modes.I, atol=1e-10 * scale)
    assert_allclose(dense.modes.V, direct.modes.V, atol=1e-10 * scale)
    assert_allclose(dense.modes.P, direct.modes.P, atol=1e-10 * scale)
    assert dense.evaluated_orders < direct.evaluated_orders


def test_turning_region_line_powers_against_independent_mpmath():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 55
    m, gamma, B = 1000, 10000, 1.0
    I, Q, V, _ = high_order_line_powers(m, gamma, B, 0.0, 0.0)
    beta = mp.sqrt(1 - mp.mpf(1) / gamma**2)
    x = m * beta  # z ~ 1: the large-order Bessel turning region
    prime = (mp.besselj(m - 1, x) - mp.besselj(m + 1, x)) / 2
    omega_b = mp.mpf(E_ESU) * B / (gamma * mp.mpf(M_E) * mp.mpf(C_CGS))
    pref = mp.mpf(E_ESU) ** 2 * omega_b**2 * m**2 / (2 * mp.pi * mp.mpf(C_CGS))
    expected = float(pref * (beta * prime) ** 2)
    assert_allclose(I, expected, rtol=1e-10)
    assert_allclose(Q, -I, rtol=1e-13)
    assert V == 0.0


def test_near_axis_stokes_v_keeps_small_nonzero_geometry_factor():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 80
    gamma, B = 1e9, 1e-8
    mu = eta = math.nextafter(1.0, 0.0)
    I, _, V, _ = high_order_line_powers(1, gamma, B, mu, eta)
    g, b, u, e = map(mp.mpf, (gamma, B, mu, eta))
    beta = mp.sqrt(1 - 1 / g**2)
    b_perp = beta * mp.sqrt(1 - u**2)
    D = 1 - beta * u * e
    x = b_perp * mp.sqrt(1 - e**2) / D
    lower, upper = mp.besselj(0, x), mp.besselj(2, x)
    a_par = (e - beta * u) * b_perp * (lower + upper) / (2 * D)
    a_perp = b_perp * (lower - upper) / 2
    omega_b = mp.mpf(E_ESU) * b / (g * mp.mpf(M_E) * mp.mpf(C_CGS))
    pref = mp.mpf(E_ESU) ** 2 * omega_b**2 / (2 * mp.pi * mp.mpf(C_CGS) * D**3)
    assert V != 0
    assert_allclose(I, float(pref * (a_par**2 + a_perp**2)), rtol=1e-13)
    assert_allclose(V, float(2 * pref * a_par * a_perp), rtol=1e-13)


def test_v_sign_and_response_edge_are_preserved():
    nu_b = _nu_b(20.0)
    channel = Channels.bump([10.0 * nu_b], [nu_b])
    edge = high_order_channel_modes(channel, 20.0, 1.0, 0.0, 0.0)
    I_10, Q_10, V_10, _ = high_order_line_powers(10, 20.0, 1.0, 0.0, 0.0)
    assert_allclose(edge.modes.I[0], I_10, rtol=1e-12)
    assert_allclose(edge.modes.P[0, 0], Q_10, rtol=1e-12)
    assert_allclose(edge.modes.V[0], V_10, atol=0)
    assert edge.active_ranges[0][0] < 9 and edge.active_ranges[0][1] > 11

    broad = Channels.bump([3000.0 * _nu_b(100)], [1000.0 * _nu_b(100)])
    a = high_order_channel_modes(broad, 100.0, 1.0, 0.3, -0.2, direct_limit=64)
    b = high_order_channel_modes(broad, 100.0, 1.0, -0.3, 0.2, direct_limit=64)
    assert_allclose(a.modes.I, b.modes.I, rtol=1e-12)
    assert_allclose(a.modes.P, b.modes.P, rtol=1e-12)
    assert_allclose(a.modes.V, -b.modes.V, rtol=1e-12)


def test_galactic_order_exploratory_value_has_finite_resource_count():
    channels = Channels.bump([1e8, 1e9, 3e9], [2e7, 2e8, 6e8])
    result = high_order_channel_modes(
        channels,
        gamma=3000.0,
        B=5e-6,
        mu=0.0,
        eta=0.0,
        phase=TaylorPhase(0),
        depth_ref=0.0,
        allow_unconverged=True,
    )
    assert min(a for a, _ in result.active_ranges) > 1e10
    assert max(b for _, b in result.active_ranges) > 1e11
    assert result.evaluated_orders < 10000
    assert result.routes == ("corrected_integral",) * 3
    assert np.all(np.isfinite(result.modes.I))
    assert np.all(result.modes.I > 0)
    assert np.all(np.isfinite(result.modes.P))
    assert result.harmonic_sum.kind == "unbounded"
    assert result.bessel.kind == result.floating_point.kind == "unbounded"
    assert not all(result.quadrature_converged)


def test_sum_strategy_uses_active_line_count_not_largest_order():
    gamma, B, mode = 3000.0, 5e-6, 10_000_000_000
    spacing = _nu_b(gamma, B)
    narrow = Channels.bump([mode * spacing], [0.4 * spacing])
    result = high_order_channel_modes(narrow, gamma, B, 0.0, 0.0, direct_limit=1)
    assert result.active_ranges[0][0] > 1e9
    assert result.routes == ("direct",)
    assert result.evaluated_orders < 16
    assert np.all(np.isfinite(result.modes.I))


def test_non_smooth_dense_response_refused_and_invalid_order_guarded():
    nu_b = _nu_b(100.0)
    channel = Channels.tophat([3000.0 * nu_b], [1000.0 * nu_b])
    with pytest.raises(ValueError, match="requires a bump"):
        high_order_channel_modes(channel, 100.0, 1.0, 0.0, 0.0, direct_limit=64)
    with pytest.raises(ValueError, match="exact unit spacing"):
        high_order_channel_modes(
            Channels.bump([1e16 * nu_b], [1e14 * nu_b]),
            100.0,
            1.0,
            0.0,
            0.0,
        )


def test_complex_inputs_rejected_before_float_cast():
    with pytest.raises(ValueError, match="real"):
        high_order_line_powers(10 + 1j, 20.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="real"):
        high_order_line_powers(10, 20.0 + 1j, 1.0, 0.0, 0.0)
    channel = Channels.bump([5.0 * _nu_b(20)], [_nu_b(20)])
    with pytest.raises(ValueError, match="real"):
        high_order_channel_modes(channel, 20.0, 1.0 + 0j, 0.0, 0.0)


@pytest.mark.parametrize("orders", [2**53, 2**53 + 1, np.array([3, 2**53 + 1])])
def test_line_powers_reject_orders_without_exact_integer_spacing(orders):
    with pytest.raises(ValueError, match="exact unit spacing"):
        high_order_line_powers(orders, 20.0, 1.0, 0.0, 0.0)


def test_nonfinite_phase_weights_are_rejected_before_channel_result():
    class NonfinitePhase:
        n_weights = 1

        def __call__(self, tau, *, depth_ref, s_depth):
            return np.full((1, *tau.shape), np.nan + 0j)

    channel = Channels.bump([5.0 * _nu_b(20)], [_nu_b(20)])
    with pytest.raises(ArithmeticError, match="nonfinite channel term"):
        high_order_channel_modes(channel, 20.0, 1.0, 0.0, 0.0, phase=NonfinitePhase())


def test_line_frequency_overflow_is_rejected_on_zero_emissivity_branch():
    with pytest.raises(ArithmeticError, match="nonfinite harmonic frequency"):
        high_order_line_powers(1, 1.0, 1e308, 0.0, 0.0)


def test_unresolved_faraday_phase_fails_closed_by_default():
    nu_b = _nu_b(20.0)
    channel = Channels.bump([300.0 * nu_b], [100.0 * nu_b])
    kwargs = dict(
        phase=TaylorPhase(1),
        depth_ref=2.0,
        s_depth=0.1,
        direct_limit=64,
        max_quad_order=128,
    )
    with pytest.raises(ArithmeticError, match="quadrature difference"):
        high_order_channel_modes(channel, 20.0, 1.0, 0.0, 0.0, **kwargs)
    exploratory = high_order_channel_modes(
        channel, 20.0, 1.0, 0.0, 0.0, allow_unconverged=True, **kwargs
    )
    assert exploratory.quadrature_converged == (False,)
    assert exploratory.harmonic_sum.kind == "unbounded"


def test_pathological_smooth_ranges_are_refused():
    nu_b = _nu_b(1.2)
    centre = (1 + 10_000_001) * nu_b / 2
    width = (10_000_001 - 1) * nu_b / 2
    planck = Channels.planck_taper([centre], [width], taper=1e-7)
    with pytest.raises(ValueError, match="requires a bump"):
        high_order_channel_modes(planck, 1.2, 1.0, 0.0, 0.0, direct_limit=64)
    bump = Channels.bump([centre], [width])
    with pytest.raises(ValueError, match="too many order scales"):
        high_order_channel_modes(bump, 1.2, 1.0, 0.0, 0.0, direct_limit=64)


def test_multichannel_table_uses_each_channels_own_response():
    nu_b = _nu_b(20.0)
    grid = np.arange(2, 9) * nu_b
    rows = np.array([[0, 0.3, 1, 0.3, 0, 0, 0], [0, 0, 0, 0.3, 1, 0.3, 0]])
    together = high_order_channel_modes(
        Channels.from_table(grid, rows, smoothness=0), 20, 1, 0, 0
    )
    for j in range(2):
        alone = high_order_channel_modes(
            Channels.from_table(grid, rows[j], smoothness=0), 20, 1, 0, 0
        )
        assert_allclose(together.modes.I[j], alone.modes.I[0], rtol=1e-13)
        assert_allclose(together.modes.P[0, j], alone.modes.P[0, 0], rtol=1e-13)


def test_line_first_derivatives_match_finite_differences_and_preserve_values():
    orders = np.array([1.0, 5.0, 20.0, 100.0])
    gamma, B, mu, eta = 20.0, 1.0, 0.3, -0.6
    value, d_gamma, d_B = _line_powers_with_derivatives(orders, gamma, B, mu, eta)
    reference = high_order_line_powers(orders, gamma, B, mu, eta)
    for actual, expected in zip(value, reference, strict=True):
        assert_allclose(actual, expected, rtol=2e-13, atol=0)
    for coordinate, step, derivative in (
        ("gamma", 1e-4, d_gamma),
        ("B", 1e-5, d_B),
    ):
        if coordinate == "gamma":
            above = high_order_line_powers(orders, gamma + step, B, mu, eta)
            below = high_order_line_powers(orders, gamma - step, B, mu, eta)
        else:
            above = high_order_line_powers(orders, gamma, B + step, mu, eta)
            below = high_order_line_powers(orders, gamma, B - step, mu, eta)
        for analytic, plus, minus in zip(derivative, above, below, strict=True):
            assert_allclose(analytic, (plus - minus) / (2 * step), rtol=3e-6)


def test_direct_channel_derivatives_agree_with_jax_low_order_kernel():
    nu_b = _nu_b(20.0)
    channels = Channels.bump([5.0 * nu_b], [2.0 * nu_b])
    phase = TaylorPhase(1)
    kwargs = dict(phase=phase, depth_ref=2e-6, s_depth=1e-7)
    result = high_order_channel_derivatives(channels, 20.0, 1.0, 0.3, -0.6, **kwargs)
    kernel = HarmonicKernel(40)
    for parameter, target in (
        ("gamma", result.d_gamma),
        ("B", result.d_B),
    ):
        for component in ("I", "V", "P"):
            if parameter == "gamma":
                expected = jax.jacfwd(
                    lambda gamma: getattr(
                        kernel.channel_modes(channels, gamma, 1.0, 0.3, -0.6, **kwargs),
                        component,
                    )
                )(20.0)
            else:
                expected = jax.jacfwd(
                    lambda B: getattr(
                        kernel.channel_modes(channels, 20.0, B, 0.3, -0.6, **kwargs),
                        component,
                    )
                )(1.0)
            assert_allclose(getattr(target, component), expected, rtol=5e-10, atol=0)
    value = high_order_channel_modes(channels, 20.0, 1.0, 0.3, -0.6, **kwargs)
    assert_allclose(result.modes.I, value.modes.I, rtol=2e-13)
    assert_allclose(result.modes.P, value.modes.P, rtol=2e-13)
    assert result.routes == ("direct",)
    assert result.quadrature.kind == "not_applicable"
    assert result.bessel.kind == result.floating_point.kind == "unbounded"


def test_dense_derivatives_match_feasible_integer_sum_with_rowwise_check():
    nu_b = _nu_b(100.0)
    channels = Channels.bump([3000.0 * nu_b], [1000.0 * nu_b])
    kwargs = dict(phase=TaylorPhase(1), depth_ref=0.01, s_depth=0.001)
    direct = high_order_channel_derivatives(
        channels, 100.0, 1.0, 0.3, 0.2, direct_limit=10000, **kwargs
    )
    dense = high_order_channel_derivatives(
        channels, 100.0, 1.0, 0.3, 0.2, direct_limit=64, **kwargs
    )
    assert direct.routes == ("direct",)
    assert dense.routes == ("corrected_integral",)
    assert dense.quadrature.kind == "estimate"
    assert np.asarray(dense.quadrature.value).shape == (3, 4, 1)
    assert dense.quadrature_converged == (True,)
    assert dense.evaluated_orders < direct.evaluated_orders
    assert dense.harmonic_sum.kind == "unbounded"
    for name in ("modes", "d_gamma", "d_B"):
        for component in ("I", "V", "P"):
            assert_allclose(
                getattr(getattr(dense, name), component),
                getattr(getattr(direct, name), component),
                rtol=1e-9,
                atol=1e-32,
            )


def test_bump_slope_edges_and_moving_harmonic_support():
    nu_b = _nu_b(20.0)
    channel = Channels.bump([10.0 * nu_b], [nu_b])
    lo, hi = np.asarray(channel.support[0])
    frequency = np.array([lo, math.nextafter(lo, hi), 10.0 * nu_b, hi])
    response, slope = _host_bump_response_and_slope(channel, 0, frequency)
    assert response[0] == response[-1] == 0
    assert slope[0] == slope[-1] == 0
    assert np.all(np.isfinite(slope))
    result = high_order_channel_derivatives(channel, 20.0, 1.0, 0.0, 0.0)
    step = 1e-4
    plus = high_order_channel_modes(channel, 20.0 + step, 1.0, 0.0, 0.0)
    minus = high_order_channel_modes(channel, 20.0 - step, 1.0, 0.0, 0.0)
    assert_allclose(
        result.d_gamma.I,
        (plus.modes.I - minus.modes.I) / (2 * step),
        rtol=1e-5,
    )


def test_turning_derivatives_use_two_neighbour_bessel_calls(monkeypatch):
    from scipy import special

    original = special.jv
    calls = []

    def counted(order, argument):
        calls.append(np.asarray(order).shape)
        return original(order, argument)

    monkeypatch.setattr(special, "jv", counted)
    result = _line_powers_with_derivatives(
        np.array([10_000_000_000.0]), 3000.0, 5e-6, 0.0, 0.0
    )
    assert len(calls) == 2
    assert all(np.all(np.isfinite(part)) for row in result for part in row)
    assert result[0][0][0] > 0


def test_low_z_fallback_and_pitch_axis_derivatives_are_finite(monkeypatch):
    from scipy import special

    original = special.jvp
    calls = []

    def counted(order, argument, derivative):
        calls.append(np.asarray(order).shape)
        return original(order, argument, derivative)

    monkeypatch.setattr(special, "jvp", counted)
    result = _line_powers_with_derivatives(
        np.array([1.0, 2.0, 3.0]), 20.0, 1.0, 0.2, 1.0 - 1e-12
    )
    assert len(calls) == 2
    assert all(np.all(np.isfinite(part)) for row in result for part in row)
    axis = high_order_channel_derivatives(
        Channels.bump([1e8], [1e7]), 20.0, 1.0, 1.0, 0.5
    )
    assert axis.routes == ("zero",)
    assert axis.evaluated_orders == 0
    assert axis.d_gamma.I[0] == axis.d_B.I[0] == 0


def test_derivative_route_rejects_unsupported_inputs_and_nonconvergence():
    nu_b = _nu_b(20.0)
    bump = Channels.bump([300.0 * nu_b], [100.0 * nu_b])
    with pytest.raises(ValueError, match="bump"):
        high_order_channel_derivatives(
            Channels.tophat([5.0 * nu_b], [nu_b]), 20.0, 1.0, 0.0, 0.0
        )
    with pytest.raises(ValueError, match="gamma > 1"):
        high_order_channel_derivatives(bump, 1.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="B > 0"):
        high_order_channel_derivatives(bump, 20.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="TaylorPhase"):
        high_order_channel_derivatives(bump, 20.0, 1.0, 0.0, 0.0, phase=object())
    with pytest.raises(ValueError, match="atol_gamma"):
        high_order_channel_derivatives(bump, 20.0, 1.0, 0.0, 0.0, atol_gamma=-1.0)
    kwargs = dict(
        phase=TaylorPhase(1),
        depth_ref=2.0,
        s_depth=0.1,
        direct_limit=64,
        max_quad_order=128,
    )
    with pytest.raises(ArithmeticError, match="derivative quadrature difference"):
        high_order_channel_derivatives(bump, 20.0, 1.0, 0.0, 0.0, **kwargs)
    exploratory = high_order_channel_derivatives(
        bump, 20.0, 1.0, 0.0, 0.0, allow_unconverged=True, **kwargs
    )
    assert exploratory.quadrature_converged == (False,)
    assert exploratory.harmonic_sum.kind == "unbounded"
