"""Finite diagnostics for the restricted whole-angle I00 prototype."""

import math

import equinox as eqx
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from syncmoments.model import Channels, TaylorPhase, high_order_channel_modes
from syncmoments.model._certified_angular_i00 import (
    AngularI00Limit,
    certify_angular_i00,
)


def _direct_gauss_i00(channels, gamma, B, order):
    """Finite independent host quadrature, not an error proof."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    positive_mu = (nodes + 1) / 2
    positive_weights = weights / 2
    total = 0.0
    for mu, w_mu in zip(positive_mu, positive_weights, strict=True):
        for eta, w_eta in zip(nodes, weights, strict=True):
            intensity = high_order_channel_modes(
                channels, gamma, B, float(mu), float(eta), direct_limit=4096
            ).modes.I[0]
            total += 0.5 * w_mu * w_eta * intensity
    return total


def test_restricted_certificate_succeeds_and_contains_finite_full_domain_checks():
    channels = Channels.bump([1e7], [1e6])
    direct_8 = _direct_gauss_i00(channels, 2.0, 1.0, 8)
    direct_16 = _direct_gauss_i00(channels, 2.0, 1.0, 16)
    assert direct_8 > 0 and direct_16 > 0
    result = certify_angular_i00(
        channels,
        2.0,
        1.0,
        direct_16,
        atol=1e-12,
        max_cells=64,
        max_evaluated_cells=128,
        max_mode_blocks=0,
    )
    assert result.absolute_error <= 1e-12
    assert result.cells <= 64
    assert result.splits > 0
    assert result.mode_blocks_evaluated == 0
    assert result.interval.a <= direct_8 <= result.interval.b
    assert result.interval.a <= direct_16 <= result.interval.b
    assert result.elapsed_seconds >= 0


def test_below_first_harmonic_has_exact_zero_full_domain():
    channels = Channels.bump([1e5], [1e4])
    result = certify_angular_i00(channels, 2.0, 1.0, 0.0, atol=1e-30, max_cells=2)
    assert result.interval == 0
    assert result.absolute_error == 0
    assert result.cells == 2
    assert dict(result.route_counts) == {"zero": 2}


def test_resource_caps_and_unsupported_inputs_fail_closed():
    channels = Channels.bump([1e7], [1e6])
    with pytest.raises(AngularI00Limit, match="cell cap.*cells=2"):
        certify_angular_i00(
            channels,
            2.0,
            1.0,
            0.0,
            atol=1e-30,
            max_cells=2,
            max_evaluated_cells=4,
            max_mode_blocks=0,
        )
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="atol"):
            certify_angular_i00(channels, 2.0, 1.0, 0.0, atol=bad)
    with pytest.raises(ValueError, match="gamma"):
        certify_angular_i00(channels, 1.0, 1.0, 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="fast_i00"):
        certify_angular_i00(channels, 2.0, 1.0, math.inf, atol=1e-12)
    with pytest.raises(ValueError, match="no phase"):
        certify_angular_i00(channels, 2.0, 1.0, 0.0, atol=1e-12, phase=TaylorPhase(0))
    with pytest.raises(ValueError, match="unit-peak bump"):
        certify_angular_i00(Channels.tophat([1e7], [1e6]), 2.0, 1.0, 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="one Channels channel"):
        certify_angular_i00(
            Channels.bump([1e7, 2e7], [1e6, 1e6]),
            2.0,
            1.0,
            0.0,
            atol=1e-12,
        )


def test_derivative_certificate_rejects_clipped_bump_support():
    channels = Channels.bump([1e8], [2e7])
    clipped = eqx.tree_at(
        lambda item: item.support,
        channels,
        jnp.asarray([[80_000_001.0, 120_000_000.0]]),
    )
    with pytest.raises(ValueError, match="contain its exact zeros"):
        certify_angular_i00(
            clipped,
            3000.0,
            5e-6,
            0.0,
            atol=1e-12,
            first_derivatives=True,
            phase=TaylorPhase(0),
        )


def test_high_gamma_broad_channel_reports_measured_cap_without_attachment():
    channels = Channels.bump([1e8], [1e7])
    # The supplied value is a finite fast 24x32 ridge diagnostic. No
    # certification is inferred from that quadrature or from this cap run.
    fast = 7.0859104450903945e-22
    with pytest.raises(AngularI00Limit, match="cell cap.*mode_blocks=") as info:
        certify_angular_i00(
            channels,
            3000.0,
            5e-6,
            fast,
            atol=1e-18,
            max_cells=256,
            max_evaluated_cells=512,
            max_mode_blocks=16,
        )
    assert "routes=" in str(info.value)


def test_high_gamma_broad_channel_can_certify_with_offline_block_budget():
    channels = Channels.bump([1e8], [1e7])
    fast = 7.0859104450903945e-22
    result = certify_angular_i00(
        channels,
        3000.0,
        5e-6,
        fast,
        atol=1e-18,
        max_cells=512,
        max_evaluated_cells=1024,
        max_mode_blocks=900,
    )
    assert result.absolute_error <= 1e-18
    assert result.mode_blocks_evaluated > 0
    assert result.mode_blocks_retained > 0
    assert result.interval.a <= fast <= result.interval.b
    with mp.workdps(100):
        lo = mp.mpf(result.interval.a._mpi_[0])
        hi = mp.mpf(result.interval.b._mpi_[1])
        binary_fast = mp.mpf(fast)
        exact_distance = max(abs(binary_fast - lo), abs(hi - binary_fast))
        assert mp.mpf(result.absolute_error) >= exact_distance
