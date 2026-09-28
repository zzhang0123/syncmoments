"""Finite overlap and failure checks for internal dense interval blocks."""

import math

import numpy as np
import pytest

from syncmoments.constants import C_CGS, E_ESU, M_E
from syncmoments.model import Channels, TaylorPhase, high_order_channel_modes
from syncmoments.model._certified_dense_harmonic import (
    _refine_blocks,
    certified_dense_channel_modes,
)


def _spacing(gamma, B, mu, eta):
    beta = math.sqrt(1 - 1 / gamma**2)
    D = 1 - beta * mu * eta
    return E_ESU * B / (2 * math.pi * gamma * M_E * C_CGS * D)


def test_dense_refinement_keeps_splittable_blocks_after_singleton_has_largest_error():
    errors = {
        (1000, 1003): 2.0,
        (1000, 1001): 0.6,
        (1002, 1003): 0.6,
        (1000, 1000): 0.6,
        (1001, 1001): 0.0,
        (1002, 1002): 0.1,
        (1003, 1003): 0.1,
    }

    def evaluate(lo, hi):
        return None, None, np.array([errors[lo, hi]])

    blocks, evaluated = _refine_blocks(
        1000, 1003, evaluate(1000, 1003), 1.0, 8, evaluate
    )
    assert evaluated == 7
    assert {(lo, hi) for _, lo, hi, _ in blocks} == {
        (1000, 1000),
        (1001, 1001),
        (1002, 1002),
        (1003, 1003),
    }
    assert sum(block[3][2][0] for block in blocks) < 1.0


def test_dense_interval_contains_feasible_direct_sum_with_v_and_phase():
    pytest.importorskip("mpmath")
    gamma, B, mu, eta = 20.0, 1.0, 0.2, 0.2
    spacing = _spacing(gamma, B, mu, eta)
    channels = Channels.bump([2000 * spacing], [200 * spacing])
    kwargs = dict(phase=TaylorPhase(1), depth_ref=0.1, s_depth=0.01)
    enclosed = certified_dense_channel_modes(
        channels, gamma, B, mu, eta, atol=1e-15, max_blocks=256, **kwargs
    )
    direct = high_order_channel_modes(
        channels, gamma, B, mu, eta, direct_limit=1000, **kwargs
    )
    assert 1 < enclosed.retained_blocks[0] < 256
    assert enclosed.evaluated_blocks[0] == 2 * enclosed.retained_blocks[0] - 1
    assert enclosed.absolute_error.kind == "bound"
    assert enclosed.harmonic_truncation.kind == "bound"
    assert direct.modes.V[0] != 0
    assert np.all(
        np.abs(enclosed.modes.I - direct.modes.I) <= enclosed.absolute_error.value[0]
    )
    assert np.all(
        np.abs(enclosed.modes.V - direct.modes.V) <= enclosed.absolute_error.value[1]
    )
    assert np.all(
        np.abs(enclosed.modes.P - direct.modes.P) <= enclosed.absolute_error.value[2:]
    )


def test_galactic_bright_channel_returns_bounded_fixed_point_value():
    pytest.importorskip("mpmath")
    channels = Channels.bump([1e8], [2e7])
    enclosed = certified_dense_channel_modes(
        channels, 3000.0, 5e-6, 0.0, 0.0, atol=1e-19, max_blocks=512
    )
    exploratory = high_order_channel_modes(
        channels, 3000.0, 5e-6, 0.0, 0.0, max_quad_order=2048
    )
    assert 1 < enclosed.retained_blocks[0] < 512
    assert enclosed.evaluated_blocks[0] == 2 * enclosed.retained_blocks[0] - 1
    assert enclosed.absolute_error.value[0, 0] <= 1e-19
    assert np.abs(enclosed.modes.I - exploratory.modes.I)[0] <= float(
        enclosed.absolute_error.value[0, 0]
    )
    assert exploratory.harmonic_sum.kind == "unbounded"
    with pytest.raises(ArithmeticError, match="exceeds max_blocks"):
        certified_dense_channel_modes(
            channels, 3000.0, 5e-6, 0.0, 0.0, atol=1e-19, max_blocks=16
        )


def test_dense_interval_rejects_unvalidated_geometry():
    pytest.importorskip("mpmath")
    channels = Channels.bump([1e8], [2e7])
    with pytest.raises(ValueError, match="Airy interval"):
        certified_dense_channel_modes(channels, 3000.0, 5e-6, 0.2, -0.1, atol=1e-19)


def test_dense_interval_accepts_distinct_channel_absolute_budgets():
    pytest.importorskip("mpmath")
    channels = Channels.bump([1e8, 3e9], [2e7, 6e8])
    result = certified_dense_channel_modes(
        channels,
        3000.0,
        5e-6,
        0.0,
        0.0,
        atol=[1e-19, 1e-23],
        max_blocks=1024,
    )
    assert np.all(result.absolute_error.value[:, 0] <= 1e-19)
    assert np.all(result.absolute_error.value[:, 1] <= 1e-23)
    with pytest.raises(ValueError, match="per channel"):
        certified_dense_channel_modes(channels, 3000.0, 5e-6, 0.0, 0.0, atol=[1e-19])
