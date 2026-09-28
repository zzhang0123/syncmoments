"""Finite overlap and guard checks for the all-z integer-line envelope."""

import math
from random import Random

import numpy as np
import pytest
from scipy.special import jv, jvp

from syncmoments.constants import C_CGS, E_ESU, M_E
from syncmoments.model import (
    Channels,
    TaylorPhase,
    high_order_channel_derivatives,
    high_order_channel_modes,
)
from syncmoments.model._angular_global_bounds import (
    angular_global_first_derivative_bounds,
    angular_global_magnitude_bounds,
    prepare_global_bound_constants,
    prepare_global_cell,
)
from syncmoments.model._interval_angular_geometry import (
    active_channel_ranges,
    angular_box_geometry,
    prepare_angular_geometry,
    prepare_channel_supports,
)
from syncmoments.model._ridge_interval_cells import (
    prepare_ridge_slab,
    ridge_interval_cell,
)


def _context():
    MPIntervalContext = pytest.importorskip("mpmath.ctx_iv").MPIntervalContext
    ctx = MPIntervalContext()
    ctx.dps = 70
    return ctx


def _spacing(gamma, B, mu, eta):
    beta = math.sqrt(1 - 1 / gamma**2)
    return E_ESU * B / (2 * math.pi * gamma * M_E * C_CGS * (1 - beta * mu * eta))


def test_all_z_bounds_cover_random_feasible_direct_channel_sums_with_phase():
    ctx = _context()
    gamma, B = 20.0, 1.0
    fixed = prepare_angular_geometry(ctx, gamma, B)
    constants = prepare_global_bound_constants(ctx)
    phase = TaylorPhase(2)
    rng = Random(810)
    for _ in range(12):
        mu, eta = rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)
        geometry = angular_box_geometry(ctx, fixed, (mu, mu), (eta, eta))
        spacing = _spacing(gamma, B, mu, eta)
        channels = Channels.bump([rng.choice((5, 10, 40)) * spacing], [2 * spacing])
        supports = prepare_channel_supports(ctx, np.asarray(channels.support))
        first, last = active_channel_ranges(ctx, supports, geometry[-1])[0]
        bound = angular_global_magnitude_bounds(
            ctx,
            geometry,
            first,
            last,
            float(channels.support[0, 0]),
            phase=phase,
            s_depth=0.01,
            constants=constants,
        )
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
        assert direct.routes == ("direct",)
        values = (direct.modes.I[0], direct.modes.V[0], *direct.modes.P[:, 0])
        for value, interval in zip(values, bound, strict=True):
            assert interval.a == 0
            assert abs(value) <= float(interval.b)


def test_exact_axis_and_z_upper_one_are_supported():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 20.0, 1.0)
    zero_perp = angular_box_geometry(ctx, fixed, (1.0, 1.0), (0.2, 0.2))
    zero_bound = angular_global_magnitude_bounds(ctx, zero_perp, 1, 1000, 1.0)
    assert all(value == 0 for value in zero_bound)
    eta_axis = angular_box_geometry(ctx, fixed, (0.2, 0.2), (1.0, 1.0))
    assert eta_axis[3].b == 0
    spacing = _spacing(20.0, 1.0, 0.2, 1.0)
    channels = Channels.bump([spacing], [0.2 * spacing])
    first, last = active_channel_ranges(
        ctx, np.asarray(channels.support), eta_axis[-1]
    )[0]
    bound = angular_global_magnitude_bounds(
        ctx, eta_axis, first, last, float(channels.support[0, 0])
    )
    direct = high_order_channel_modes(channels, 20.0, 1.0, 0.2, 1.0, direct_limit=4096)
    assert bound[0].b > 0
    assert abs(direct.modes.I[0]) <= float(bound[0].b)
    broad = angular_box_geometry(ctx, fixed, (0.2, 0.8), (0.2, 0.8))
    widened = (*broad[:3], ctx.mpf([broad[3].a, 1]), *broad[4:])
    assert widened[3].b == 1
    assert angular_global_magnitude_bounds(ctx, widened, 1, 20, 1.0)[0].b > 0


def test_coarse_first_derivative_bounds_cover_direct_interior_points():
    ctx = _context()
    gamma, B = 2.0, 1.0
    channels = Channels.bump([1e7], [1e6])
    fixed = prepare_angular_geometry(ctx, gamma, B)
    constants = prepare_global_bound_constants(ctx)
    geometry = angular_box_geometry(ctx, fixed, (0.18, 0.22), (0.28, 0.32))
    first, last = active_channel_ranges(
        ctx, prepare_channel_supports(ctx, np.asarray(channels.support)), geometry[-1]
    )[0]
    gamma_bound, B_bound = angular_global_first_derivative_bounds(
        ctx,
        geometry,
        fixed,
        first,
        last,
        float(channels.support[0, 0]),
        float(channels.support[0, 1]),
        float(channels.widths_hz[0]),
        B,
        0.22,
        0.32,
        depth_ref=1e-6,
        s_depth=0.2,
        max_b=1,
        constants=constants,
    )
    direct = high_order_channel_derivatives(
        channels,
        gamma,
        B,
        0.2,
        0.3,
        phase=TaylorPhase(1),
        depth_ref=1e-6,
        s_depth=0.2,
    )
    assert direct.routes == ("direct",)
    for modes, bound in ((direct.d_gamma, gamma_bound), (direct.d_B, B_bound)):
        assert abs(modes.I[0]) <= float(bound[0].b)
        assert abs(modes.V[0]) <= float(bound[0].b)
        for degree in range(2):
            assert abs(modes.P[degree, 0]) <= float(bound[degree + 1].b)


def test_prepared_cell_matches_raw_geometry_and_empty_range_is_zero():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 20.0, 1.0)
    geometry = angular_box_geometry(ctx, fixed, (0.2, 0.4), (0.1, 0.3))
    constants = prepare_global_bound_constants(ctx)
    cell = prepare_global_cell(ctx, geometry, constants=constants)
    phase = TaylorPhase(2)
    for first, last, frequency_lo in ((1, 9, 1e5), (100, 1000, 1e6)):
        raw = angular_global_magnitude_bounds(
            ctx,
            geometry,
            first,
            last,
            frequency_lo,
            phase=phase,
            s_depth=0.01,
            constants=constants,
        )
        prepared = angular_global_magnitude_bounds(
            ctx,
            cell,
            first,
            last,
            frequency_lo,
            phase=phase,
            s_depth=0.01,
        )
        assert prepared == raw
        assert len(prepared) == 5
    empty = angular_global_magnitude_bounds(ctx, cell, 0, -1, 1.0, phase=phase)
    assert len(empty) == 5
    assert all(value == 0 for value in empty)


def test_correlated_near_ridge_cell_encloses_direct_interior_point():
    ctx = _context()
    gamma, B, mu, u = 20.0, 1.0, 0.5, 0.1
    fixed = prepare_angular_geometry(ctx, gamma, B)
    slab = prepare_ridge_slab(ctx, fixed, (0.49, 0.51))
    geometry = ridge_interval_cell(ctx, slab, 1, (0.0, 0.2)).geometry
    eta = mu + slab.h * math.sinh(u * math.asinh((1 - mu) / slab.h))
    spacing = _spacing(gamma, B, mu, eta)
    channels = Channels.bump([10 * spacing], [2 * spacing])
    first, last = active_channel_ranges(
        ctx, np.asarray(channels.support), geometry[-1]
    )[0]
    bound = angular_global_magnitude_bounds(
        ctx, geometry, first, last, float(channels.support[0, 0])
    )
    direct = high_order_channel_modes(channels, gamma, B, mu, eta, direct_limit=4096)
    assert geometry[3].b > ctx.mpf("0.9")
    assert abs(direct.modes.I[0]) <= float(bound[0].b)


def test_all_z_order_decay_and_axis_prefactor_caps_contain_direct_high_orders():
    ctx = _context()
    gamma, B, mu, eta = 3000.0, 5e-6, 0.5, 0.5
    fixed = prepare_angular_geometry(ctx, gamma, B)
    geometry = angular_box_geometry(ctx, fixed, (mu, mu), (eta, eta))
    beta = math.sqrt(1 - gamma**-2)
    D = 1 - beta * mu * eta
    perpendicular = beta * math.sqrt(1 - mu**2)
    z = perpendicular * math.sqrt(1 - eta**2) / D
    omega_b = E_ESU * B / (gamma * M_E * C_CGS)
    for order in (1, 10, 1000, 100_000):
        parallel = (eta - beta * mu) * perpendicular * jv(order, order * z) / (D * z)
        transverse = perpendicular * jvp(order, order * z)
        direct = (
            E_ESU**2
            * omega_b**2
            * order**2
            / (2 * math.pi * C_CGS * D**3)
            * (parallel**2 + transverse**2)
        )
        bound = angular_global_magnitude_bounds(ctx, geometry, order, order, 1.0)[0]
        assert direct <= float(bound.b)
    axis_box = angular_box_geometry(ctx, fixed, (0.998, 1.0), (0.999, 1.0))
    plain = prepare_global_cell(ctx, axis_box)
    physical = prepare_global_cell(ctx, axis_box, fixed=fixed, mu_abs_lower=0.998)
    assert physical.intensity_base_upper <= plain.intensity_base_upper


def test_uniform_derivative_two_thirds_bound_samples_and_crossover():
    ctx = _context()
    constants = prepare_global_bound_constants(ctx)
    C = constants.derivative_two_thirds_constant
    assert float(C.a) > 2 / math.pi
    for order in (1, 2, 42, 43, 1000, 100_000):
        envelope = float(C.b) * order ** (-2 / 3) * (1 + 1e-12)
        for z in (0.0, 0.2, 0.8, 0.99, 1.0):
            assert abs(jvp(order, order * z)) <= envelope
    for order in (42, 43, 1000):
        old = (constants.derivative_order_constant**2 * order).b
        new = (
            constants.derivative_two_thirds_constant**2 * ctx.exp(2 * ctx.ln(order) / 3)
        ).b
        assert (new > old) == (order == 42)


def test_transverse_order_envelope_increases_and_preserves_subfloat_values():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 3000.0, 5e-6)
    geometry = angular_box_geometry(ctx, fixed, (0.0, 0.0), (0.0, 0.0))
    cell = prepare_global_cell(ctx, geometry)
    assert cell.parallel_base_upper == 0
    orders = (1, 2, 42, 43, 1000, 100_000)
    values = [
        angular_global_magnitude_bounds(ctx, cell, order, order, 1.0)[0].b
        for order in orders
    ]
    assert values == sorted(values)
    old_high_order = (
        cell.transverse_base_upper * cell.derivative_order_constant**2 * orders[-1]
    ).b
    assert values[-1] < old_high_order
    tiny = prepare_angular_geometry(ctx, 20.0, 1e-160)
    tiny_geometry = angular_box_geometry(ctx, tiny, (0.0, 0.0), (0.0, 0.0))
    tiny_bound = angular_global_magnitude_bounds(
        ctx, tiny_geometry, 100_000, 100_000, 1.0
    )[0]
    assert tiny_bound.b > 0
    assert float(tiny_bound.b) == 0.0


def test_rejects_malformed_inputs_context_and_unrepresentable_output():
    ctx = _context()
    other = _context()
    fixed = prepare_angular_geometry(ctx, 20.0, 1.0)
    geometry = angular_box_geometry(ctx, fixed, (0.2, 0.2), (0.3, 0.3))
    for first, last in ((-1, 10), (2, 1), (1, 2**53), (False, -1), (0.0, -1)):
        with pytest.raises(ValueError, match="first"):
            angular_global_magnitude_bounds(ctx, geometry, first, last, 1.0)
    with pytest.raises(ValueError, match="positive"):
        angular_global_magnitude_bounds(ctx, geometry, 1, 10, 0)
    with pytest.raises(ValueError, match="TaylorPhase"):
        angular_global_magnitude_bounds(ctx, geometry, 1, 10, 1.0, phase=object())
    with pytest.raises(ValueError, match="positive"):
        angular_global_magnitude_bounds(ctx, geometry, 1, 10, 1.0, s_depth=-1)
    bad_z = (*geometry[:3], ctx.mpf([0, 1.1]), *geometry[4:])
    with pytest.raises(ValueError, match="0 <= z <= 1"):
        prepare_global_cell(ctx, bad_z)
    bad_D = (ctx.mpf([0, 1]), *geometry[1:])
    with pytest.raises(ValueError, match="positive geometry"):
        prepare_global_cell(ctx, bad_D)
    with pytest.raises(ValueError, match="interval context"):
        prepare_global_cell(
            ctx, geometry, constants=prepare_global_bound_constants(other)
        )
    cell = prepare_global_cell(ctx, geometry)
    with pytest.raises(ValueError, match="prepared cell"):
        angular_global_magnitude_bounds(other, cell, 1, 10, 1.0)
    huge_omega = (*geometry[:4], ctx.mpf("1e300"), geometry[5])
    with pytest.raises(ArithmeticError, match="representable"):
        prepare_global_cell(ctx, huge_omega)
