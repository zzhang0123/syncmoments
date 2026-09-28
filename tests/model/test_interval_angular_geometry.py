"""Finite probes of conditional interval angular geometry enclosures."""

import math

import pytest

from syncmoments.constants import C_CGS, E_ESU, M_E
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


def _point_geometry(ctx, gamma, B, mu, eta):
    g, field, pitch, view = map(ctx.mpf, (gamma, B, mu, eta))
    beta = ctx.sqrt(1 - 1 / g**2)
    D = 1 - beta * pitch * view
    delta = view - beta * pitch
    perpendicular = beta * ctx.sqrt(1 - pitch**2)
    z = perpendicular * ctx.sqrt(1 - view**2) / D
    omega_b = ctx.mpf(E_ESU) * field / (g * ctx.mpf(M_E) * ctx.mpf(C_CGS))
    return D, delta, perpendicular, z, omega_b, omega_b / (2 * ctx.pi * D)


@pytest.mark.parametrize(
    ("gamma", "B", "mu_box", "eta_box", "samples"),
    [
        (20.0, 1.0, (0.2, 0.5), (0.1, 0.6), ((0.3, 0.2), (0.4, 0.5))),
        (20.0, 1.0, (-0.6, -0.2), (-0.5, -0.1), ((-0.4, -0.2),)),
        (20.0, 1.0, (-0.3, 0.2), (-0.4, 0.3), ((0.1, -0.2), (-0.2, 0.1))),
        (
            1e9,
            1e-8,
            (1 - 2e-12, 1.0),
            (1 - 3e-12, 1.0),
            ((1 - 1e-12, 1 - 1e-12),),
        ),
        (20.0, 1.0, (-1.0, -0.8), (0.8, 1.0), ((-0.9, 0.9),)),
    ],
)
def test_angular_box_encloses_independent_point_geometry(
    gamma, B, mu_box, eta_box, samples
):
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, gamma, B)
    box = angular_box_geometry(ctx, fixed, mu_box, eta_box)
    assert box[0].a > 0
    assert box[3].a >= 0 and box[3].b <= 1
    assert box[-1].a > 0
    for mu, eta in samples:
        point = _point_geometry(ctx, gamma, B, mu, eta)
        for index, (enclosure, value) in enumerate(zip(box, point, strict=True)):
            assert enclosure.a <= value.a, index
            assert enclosure.b >= value.b, index


def test_axis_boxes_are_valid_and_expose_zero_z():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 30.0, 1.0)
    for mu_box, eta_box in [((0.8, 1.0), (0.7, 1.0)), ((-1.0, -0.8), (-1.0, -0.7))]:
        D, _, perpendicular, z, _, spacing = angular_box_geometry(
            ctx, fixed, mu_box, eta_box
        )
        assert D.a > 0 and spacing.a > 0
        assert perpendicular.a == 0
        assert z.a == 0


def test_defect_identity_tightens_beaming_ridge_without_losing_points():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 3000.0, 5e-6)
    box = angular_box_geometry(ctx, fixed, (0.49995, 0.50005), (0.49995, 0.50005))
    z = box[3]
    assert z.a > ctx.mpf("0.9999999")
    assert z.b <= 1
    for mu, eta in [(0.49996, 0.50004), (0.5, 0.5), (0.50004, 0.49996)]:
        point_z = _point_geometry(ctx, 3000.0, 5e-6, mu, eta)[3]
        assert z.a <= point_z.a
        assert z.b >= point_z.b


def test_positive_defect_lower_endpoint_tightens_z_even_when_upper_exceeds_one():
    ctx = _context()
    gamma, B = 3000.0, 5e-6
    fixed = prepare_angular_geometry(ctx, gamma, B)
    mu_box = (0.998046875, 0.9990234375)
    eta_box = (0.9994, 0.9997623)
    box = angular_box_geometry(ctx, fixed, mu_box, eta_box)
    D, delta, _, z, _, _ = box
    eta = ctx.mpf(eta_box)
    sine = ctx.sqrt((1 - eta) * (1 + eta))
    defect = (delta / D) ** 2 + (fixed.inverse_gamma * sine / D) ** 2
    assert defect.a > 0 and defect.b > 1
    assert z.b < ctx.mpf("0.99")
    for mu in mu_box:
        for view in eta_box:
            point_z = _point_geometry(ctx, gamma, B, mu, view)[3]
            assert z.a <= point_z.a <= point_z.b <= z.b


def test_active_ranges_cover_every_sampled_cell_order_and_keep_far_channels_disjoint():
    ctx = _context()
    gamma, B = 20.0, 1.0
    fixed = prepare_angular_geometry(ctx, gamma, B)
    geometry = angular_box_geometry(ctx, fixed, (-0.2, 0.3), (-0.15, 0.35))
    central_spacing = float((fixed.omega_b / (2 * ctx.pi)).mid)
    supports = (
        (8.1 * central_spacing, 12.9 * central_spacing),
        (100_000.1 * central_spacing, 100_100.9 * central_spacing),
    )
    ranges = active_channel_ranges(ctx, supports, geometry[-1])
    prepared = prepare_channel_supports(ctx, supports)
    assert active_channel_ranges(ctx, prepared, geometry[-1]) == ranges
    assert ranges[0][1] < ranges[1][0]
    for mu, eta in [(-0.18, 0.3), (0.0, 0.0), (0.28, -0.12), (0.2, 0.3)]:
        spacing = float(_point_geometry(ctx, gamma, B, mu, eta)[-1].mid)
        for (lo, hi), (first, last) in zip(supports, ranges, strict=True):
            first_active = math.floor(lo / spacing) + 1
            last_active = math.ceil(hi / spacing) - 1
            if first_active <= last_active:
                assert first <= first_active
                assert last >= last_active


def test_empty_support_below_first_line_and_guard_large_orders():
    ctx = _context()
    assert active_channel_ranges(ctx, ((0.1, 0.2),), ctx.mpf([1, 2])) == ((0, -1),)
    with pytest.raises(ValueError, match="exact float-order limit"):
        active_channel_ranges(ctx, ((1e16, 1.01e16),), ctx.mpf(1))


@pytest.mark.parametrize(
    ("mu_box", "eta_box"),
    [
        ((0.2, 0.1), (0.0, 0.1)),
        ((0.0, 1.1), (0.0, 0.1)),
        ((0.0, 0.1), ()),
        ((float("nan"), 0.1), (0.0, 0.1)),
    ],
)
def test_rejects_malformed_angular_boxes(mu_box, eta_box):
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 20.0, 1.0)
    with pytest.raises(ValueError):
        angular_box_geometry(ctx, fixed, mu_box, eta_box)


def test_rejects_context_mismatch_and_unusable_spacing():
    ctx = _context()
    other = _context()
    fixed = prepare_angular_geometry(ctx, 20.0, 1.0)
    with pytest.raises(ValueError, match="supplied interval context"):
        angular_box_geometry(other, fixed, (0.0, 0.1), (0.0, 0.1))
    with pytest.raises(ValueError, match="positive interval"):
        active_channel_ranges(ctx, ((1.0, 2.0),), ctx.mpf([0, 1]))
    with pytest.raises(ValueError, match="padding"):
        active_channel_ranges(ctx, ((1.0, 2.0),), ctx.mpf(1), padding=1)
    prepared = prepare_channel_supports(ctx, ((1.0, 2.0),))
    with pytest.raises(ValueError, match="prepared supports"):
        active_channel_ranges(other, prepared, ctx.mpf(1))
