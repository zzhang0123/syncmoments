"""Finite overlap and guard checks for conditional off-ridge cell bounds."""

import math

import numpy as np
import pytest

from syncmoments.constants import C_CGS, E_ESU, M_E
from syncmoments.model import Channels, TaylorPhase, high_order_channel_modes
from syncmoments.model._angular_faint_bounds import (
    angular_faint_magnitude_bounds,
    prepare_faint_bound_constants,
    prepare_faint_cell,
)
from syncmoments.model._interval_angular_geometry import (
    active_channel_ranges,
    angular_box_geometry,
    prepare_angular_geometry,
    prepare_channel_supports,
)


def _context():
    MPIntervalContext = pytest.importorskip("mpmath.ctx_iv").MPIntervalContext
    ctx = MPIntervalContext()
    ctx.dps = 70
    return ctx


def _point_spacing(gamma, B, mu, eta):
    beta = math.sqrt(1 - 1 / gamma**2)
    D = 1 - beta * mu * eta
    return E_ESU * B / (2 * math.pi * gamma * M_E * C_CGS * D)


def test_off_ridge_bounds_cover_sampled_direct_sums_in_disjoint_channels():
    ctx = _context()
    gamma, B, mu, eta = 20.0, 1.0, 0.8, -0.7
    fixed = prepare_angular_geometry(ctx, gamma, B)
    geometry = angular_box_geometry(ctx, fixed, (mu, mu), (eta, eta))
    spacing = _point_spacing(gamma, B, mu, eta)
    channels = Channels.bump([8 * spacing, 42 * spacing], [2 * spacing] * 2)
    supports = prepare_channel_supports(ctx, np.asarray(channels.support))
    ranges = active_channel_ranges(ctx, supports, geometry[-1])
    assert ranges[0][1] < ranges[1][0]
    phase = TaylorPhase(2)
    constants = prepare_faint_bound_constants(ctx)
    prepared_cell = prepare_faint_cell(ctx, geometry, constants=constants)
    bounds = [
        angular_faint_magnitude_bounds(
            ctx,
            geometry,
            first,
            last,
            float(channels.support[j, 0]),
            phase=phase,
            s_depth=0.01,
            constants=constants,
        )
        for j, (first, last) in enumerate(ranges)
    ]
    prepared_bounds = [
        angular_faint_magnitude_bounds(
            ctx,
            prepared_cell,
            first,
            last,
            float(channels.support[j, 0]),
            phase=phase,
            s_depth=0.01,
        )
        for j, (first, last) in enumerate(ranges)
    ]
    assert bounds == prepared_bounds
    direct = high_order_channel_modes(
        channels,
        gamma,
        B,
        mu,
        eta,
        phase=phase,
        depth_ref=0.2,
        s_depth=0.01,
        direct_limit=4096,
    )
    assert direct.routes == ("direct", "direct")
    for j, envelope in enumerate(bounds):
        values = (direct.modes.I[j], direct.modes.V[j], *direct.modes.P[:, j])
        for value, bound in zip(values, envelope, strict=True):
            assert abs(value) <= float(bound.b)
            assert bound.a == 0


@pytest.mark.parametrize(("mu", "eta"), [(1.0, 0.3), (0.2, 1.0), (-1.0, -0.4)])
def test_exact_axis_boxes_have_zero_bound_for_orders_at_least_three(mu, eta):
    ctx = _context()
    gamma, B = 20.0, 1.0
    fixed = prepare_angular_geometry(ctx, gamma, B)
    geometry = angular_box_geometry(ctx, fixed, (mu, mu), (eta, eta))
    assert geometry[3].b == 0
    bounds = angular_faint_magnitude_bounds(
        ctx, geometry, 3, 100_000, 1.0, phase=TaylorPhase(2), s_depth=0.1
    )
    assert len(bounds) == 5
    assert all(bound == 0 for bound in bounds)
    lines = high_order_channel_modes(
        Channels.bump(
            [5 * _point_spacing(gamma, B, mu, eta)],
            [0.5 * _point_spacing(gamma, B, mu, eta)],
        ),
        gamma,
        B,
        mu,
        eta,
        direct_limit=4096,
    )
    assert lines.modes.I[0] == 0


def test_axis_touching_box_with_positive_z_upper_bounds_interior_point():
    ctx = _context()
    gamma, B, mu, eta = 20.0, 1.0, 0.995, 0.35
    fixed = prepare_angular_geometry(ctx, gamma, B)
    geometry = angular_box_geometry(ctx, fixed, (0.99, 1.0), (0.3, 0.4))
    assert geometry[3].a == 0
    assert 0 < geometry[3].b < ctx.mpf("0.9")
    spacing = _point_spacing(gamma, B, mu, eta)
    channels = Channels.bump([8 * spacing], [2 * spacing])
    first, last = active_channel_ranges(
        ctx, np.asarray(channels.support), geometry[-1]
    )[0]
    bound = angular_faint_magnitude_bounds(
        ctx, geometry, first, last, float(channels.support[0, 0])
    )
    direct = high_order_channel_modes(channels, gamma, B, mu, eta, direct_limit=4096)
    assert bound[0].b > 0
    assert abs(direct.modes.I[0]) <= float(bound[0].b)


def test_monotonic_endpoint_envelopes_and_zero_axis_limit():
    ctx = _context()
    m = 3
    values = []
    derivative_values = []
    for z in (1e-5, 0.001, 0.05, 0.2, 0.6, 0.9):
        z = ctx.mpf(z)
        root = ctx.sqrt((1 - z) * (1 + z))
        xi = ctx.ln((1 + root) / z) - root
        decay = ctx.exp(-m * xi)
        values.append(float((decay / z).mid))
        derivative_values.append(
            float(
                (
                    ctx.sqrt(ctx.sqrt(1 + z**2))
                    * decay
                    / (z * ctx.sqrt(2 * ctx.pi * m))
                ).mid
            )
        )
    assert values == sorted(values)
    assert derivative_values == sorted(derivative_values)
    assert values[0] < 1e-8


def test_rejects_nonfaint_geometry_invalid_ranges_or_unrepresentable_magnitude():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 20.0, 1.0)
    geometry = angular_box_geometry(ctx, fixed, (0.8, 0.8), (-0.7, -0.7))
    bad_z = (*geometry[:3], ctx.mpf([0.8, 1.0]), *geometry[4:])
    near_turn = (*geometry[:3], ctx.mpf([0.9, 0.95]), *geometry[4:])
    bad_D = (ctx.mpf([0, 1]), *geometry[1:])
    for args in [(2, 10), (5, 4), (3, 2**53)]:
        with pytest.raises(ValueError, match="first"):
            angular_faint_magnitude_bounds(ctx, geometry, *args, 1.0)
    with pytest.raises(ValueError, match="0 <= z < 1"):
        angular_faint_magnitude_bounds(ctx, bad_z, 3, 10, 1.0)
    with pytest.raises(ValueError, match="monotonicity guard"):
        angular_faint_magnitude_bounds(ctx, near_turn, 3, 10, 1.0)
    assert angular_faint_magnitude_bounds(ctx, near_turn, 4, 10, 1.0)[0].b > 0
    with pytest.raises(ValueError, match="positive geometry"):
        angular_faint_magnitude_bounds(ctx, bad_D, 3, 10, 1.0)
    with pytest.raises(ValueError):
        angular_faint_magnitude_bounds(ctx, geometry, 3, 10, 0)
    for kwargs in [dict(s_depth=-1), dict(phase=object())]:
        with pytest.raises(ValueError):
            angular_faint_magnitude_bounds(ctx, geometry, 3, 10, 1.0, **kwargs)
    huge_omega = (*geometry[:4], ctx.mpf("1e300"), geometry[5])
    with pytest.raises(ArithmeticError, match="representable"):
        angular_faint_magnitude_bounds(ctx, huge_omega, 3, 10, 1.0)


def test_rejects_precomputed_constants_from_another_context():
    ctx = _context()
    other = _context()
    fixed = prepare_angular_geometry(ctx, 20.0, 1.0)
    geometry = angular_box_geometry(ctx, fixed, (0.8, 0.8), (-0.7, -0.7))
    with pytest.raises(ValueError, match="supplied interval context"):
        angular_faint_magnitude_bounds(
            ctx, geometry, 3, 10, 1.0, constants=prepare_faint_bound_constants(other)
        )
    prepared_cell = prepare_faint_cell(ctx, geometry)
    with pytest.raises(ValueError, match="prepared cell"):
        angular_faint_magnitude_bounds(other, prepared_cell, 3, 10, 1.0)


def test_geometric_order_tail_tightens_broad_faint_range():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 3000.0, 5e-6)
    geometry = angular_box_geometry(ctx, fixed, (0.5, 0.5), (0.55, 0.55))
    prepared = prepare_faint_cell(ctx, geometry)
    first, last = 1000, 100_000_000
    bound = angular_faint_magnitude_bounds(ctx, prepared, first, last, 1.0)[0]
    decay = ctx.exp(-first * prepared.xi_lower).b
    old_count_bound = (
        (last - first + 1)
        * prepared.prefactor_base_upper
        * last**2
        * (
            (prepared.parallel_base_upper * decay) ** 2
            + (prepared.derivative_base_upper * decay / ctx.sqrt(first)) ** 2
        )
    ).b
    assert bound.b < old_count_bound
    axis_box = angular_box_geometry(ctx, fixed, (0.998, 1.0), (0.8, 0.9))
    if axis_box[3].b < 1:
        plain = prepare_faint_cell(ctx, axis_box)
        physical = prepare_faint_cell(ctx, axis_box, fixed=fixed, mu_abs_lower=0.998)
        assert physical.prefactor_base_upper <= plain.prefactor_base_upper
