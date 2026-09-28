"""Finite overlap checks for the internal sparse interval channel route."""

import math

import numpy as np
import pytest

from syncmoments.constants import C_CGS, E_ESU, M_E
from syncmoments.model import Channels, TaylorPhase, high_order_channel_modes
from syncmoments.model._certified_sparse_harmonic import certified_sparse_channel_modes
from syncmoments.model.high_order_harmonic import high_order_line_powers


def _spacing(gamma, B):
    return E_ESU * B / (2 * math.pi * gamma * M_E * C_CGS)


def test_sparse_interval_encloses_direct_channels_with_line_phase():
    pytest.importorskip("mpmath")
    gamma, B = 20.0, 1.0
    spacing = _spacing(gamma, B)
    channels = Channels.bump([5 * spacing, 10 * spacing], [0.4 * spacing] * 2)
    kwargs = dict(phase=TaylorPhase(1), depth_ref=0.1, s_depth=0.01)
    enclosed = certified_sparse_channel_modes(
        channels, gamma, B, 0.0, 0.0, max_modes=1, atol=1e-19, **kwargs
    )
    direct = high_order_channel_modes(channels, gamma, B, 0.0, 0.0, **kwargs)
    assert enclosed.evaluated_orders == 2
    assert enclosed.absolute_error.kind == "bound"
    assert enclosed.harmonic_sum.kind == "bound"
    assert np.all(np.asarray(enclosed.harmonic_sum.value) == 0)
    envelope = np.asarray(enclosed.absolute_error.value)
    assert np.all(np.abs(enclosed.modes.I - direct.modes.I) <= envelope[0])
    assert np.all(np.abs(enclosed.modes.V - direct.modes.V) <= envelope[1])
    assert np.all(np.abs(enclosed.modes.P - direct.modes.P) <= envelope[2:])
    with pytest.raises(ArithmeticError, match="requested atol"):
        certified_sparse_channel_modes(
            channels, gamma, B, 0.0, 0.0, max_modes=1, atol=0.0, **kwargs
        )


def test_sparse_interval_refuses_dense_or_unvalidated_response():
    pytest.importorskip("mpmath")
    spacing = _spacing(20.0, 1.0)
    dense = Channels.bump([100 * spacing], [20 * spacing])
    with pytest.raises(ValueError, match="exceeds max_modes"):
        certified_sparse_channel_modes(dense, 20.0, 1.0, 0.0, 0.0)
    integral = Channels.bump(
        [5 * spacing], [0.4 * spacing], normalisation="unit_integral"
    )
    with pytest.raises(ValueError, match="unit_peak"):
        certified_sparse_channel_modes(integral, 20.0, 1.0, 0.0, 0.0)


def test_sparse_interval_encloses_asymmetric_stokes_v():
    pytest.importorskip("mpmath")
    gamma, B, mu, eta = 20.0, 1.0, 0.3, -0.2
    beta = math.sqrt(1 - 1 / gamma**2)
    spacing = _spacing(gamma, B) / (1 - beta * mu * eta)
    channels = Channels.bump([5 * spacing], [0.4 * spacing])
    enclosed = certified_sparse_channel_modes(channels, gamma, B, mu, eta)
    direct = high_order_channel_modes(channels, gamma, B, mu, eta)
    assert direct.modes.V[0] != 0
    assert np.abs(enclosed.modes.V - direct.modes.V)[0] <= float(
        enclosed.absolute_error.value[1, 0]
    )


def test_sparse_interval_preserves_near_axis_stokes_v():
    pytest.importorskip("mpmath")
    gamma, B = 1e9, 1e-8
    mu = eta = math.nextafter(1.0, 0.0)
    *_, frequency = high_order_line_powers(1, gamma, B, mu, eta)
    channels = Channels.bump([float(frequency)], [float(frequency) * 1e-6])
    enclosed = certified_sparse_channel_modes(channels, gamma, B, mu, eta, max_modes=1)
    direct = high_order_channel_modes(channels, gamma, B, mu, eta)
    assert direct.modes.V[0] != 0
    assert abs(enclosed.modes.V[0] - direct.modes.V[0]) <= float(
        enclosed.absolute_error.value[1, 0]
    )


def test_dense_faint_channel_uses_whole_range_bound_without_bessel_calls():
    pytest.importorskip("mpmath")
    gamma, B, order = 2.0, 1.0, 1000
    spacing = _spacing(gamma, B)
    channel = Channels.bump([order * spacing], [20 * spacing])
    kwargs = dict(phase=TaylorPhase(1), depth_ref=0.1, s_depth=0.01)
    with pytest.raises(ValueError, match="exceeds max_modes"):
        certified_sparse_channel_modes(channel, gamma, B, 0.0, 0.0, **kwargs)
    enclosed = certified_sparse_channel_modes(
        channel, gamma, B, 0.0, 0.0, atol=2e-54, **kwargs
    )
    direct = high_order_channel_modes(
        channel, gamma, B, 0.0, 0.0, direct_limit=64, **kwargs
    )
    assert enclosed.routes == ("bounded_zero",)
    assert enclosed.evaluated_orders == 0
    assert np.all(enclosed.absolute_error.value > 0)
    assert np.all(enclosed.absolute_error.value < 2e-54)
    assert np.all(np.abs(direct.modes.I) <= enclosed.absolute_error.value[0])
    assert np.all(np.abs(direct.modes.V) <= enclosed.absolute_error.value[1])
    assert np.all(np.abs(direct.modes.P) <= enclosed.absolute_error.value[2:])
    assert np.all(enclosed.harmonic_sum.value > 0)
    with pytest.raises(ValueError, match="bound exceeds requested atol"):
        certified_sparse_channel_modes(
            channel, gamma, B, 0.0, 0.0, atol=1e-55, **kwargs
        )


def test_high_order_narrow_channel_selects_bounded_airy_line():
    pytest.importorskip("mpmath")
    gamma, B, order = 3000.0, 5e-6, 10_000_000_000
    spacing = _spacing(gamma, B)
    channel = Channels.bump([order * spacing], [0.4 * spacing])
    enclosed = certified_sparse_channel_modes(channel, gamma, B, 0.0, 0.0, max_modes=1)
    direct = high_order_channel_modes(channel, gamma, B, 0.0, 0.0, direct_limit=1)
    assert enclosed.routes == ("direct",)
    assert enclosed.bessel_methods == ((order, "airy"),)
    assert enclosed.evaluated_orders == 1
    assert abs(enclosed.modes.I[0] - direct.modes.I[0]) <= float(
        enclosed.absolute_error.value[0, 0]
    )
