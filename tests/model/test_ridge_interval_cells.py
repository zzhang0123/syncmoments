"""Finite containment and map-coverage checks for correlated ridge cells."""

import math
from random import Random

import mpmath as mp
import numpy as np
import pytest

from syncmoments.constants import C_CGS, E_ESU, M_E
from syncmoments.model._interval_angular_geometry import (
    angular_box_geometry,
    prepare_angular_geometry,
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


def _mp_endpoint(interval, upper):
    return mp.mpf(interval.b._mpi_[1] if upper else interval.a._mpi_[0])


def _point(gamma, B, h, mu, side, u):
    with mp.workdps(100):
        g, field, scale, pitch, mapped = map(mp.mpf, (gamma, B, h, mu, u))
        beta = mp.sqrt(1 - 1 / g**2)
        L = mp.asinh((1 - side * pitch) / scale)
        eta = mp.mpf(side) if u == 1 else pitch + side * scale * mp.sinh(mapped * L)
        jacobian = scale * L * mp.cosh(mapped * L)
        D = 1 - beta * pitch * eta
        delta = eta - beta * pitch
        perpendicular = beta * mp.sqrt(1 - pitch**2)
        z = perpendicular * mp.sqrt(1 - eta**2) / D
        omega_b = mp.mpf(E_ESU) * field / (g * mp.mpf(M_E) * mp.mpf(C_CGS))
        geometry = (D, delta, perpendicular, z, omega_b, omega_b / (2 * mp.pi * D))
        return geometry, eta, jacobian


def _contains(interval, point):
    with mp.workdps(100):
        assert _mp_endpoint(interval, False) <= point
        assert point <= _mp_endpoint(interval, True)


@pytest.mark.parametrize("side", (-1, 1))
@pytest.mark.parametrize("u_box", ((0.0, 0.1), (0.25, 0.55), (0.9, 1.0)))
def test_mapped_cell_contains_high_precision_points_including_corners(side, u_box):
    ctx = _context()
    gamma, B = 3000.0, 5e-6
    fixed = prepare_angular_geometry(ctx, gamma, B)
    slab = prepare_ridge_slab(ctx, fixed, (0.49995, 0.50005))
    cell = ridge_interval_cell(ctx, slab, side, u_box)
    assert cell.geometry[0].a > 0
    assert cell.geometry[3].a >= 0 and cell.geometry[3].b <= 1
    assert cell.jacobian.a >= 0
    for mu in (slab.mu_lo, (slab.mu_lo + slab.mu_hi) / 2, slab.mu_hi):
        for u in (u_box[0], (u_box[0] + u_box[1]) / 2, u_box[1]):
            geometry, eta, jacobian = _point(gamma, B, slab.h, mu, side, u)
            for interval, point in zip(cell.geometry, geometry, strict=True):
                _contains(interval, point)
            _contains(cell.eta, eta)
            _contains(cell.jacobian, jacobian)


def test_deterministic_random_point_containment_across_positive_mu_slabs():
    ctx = _context()
    rng = Random(173)
    for gamma, B in ((2.0, 1.0), (3000.0, 5e-6), (1e9, 1e-8)):
        fixed = prepare_angular_geometry(ctx, gamma, B)
        for _ in range(20):
            a = rng.uniform(0, 0.9)
            b = rng.uniform(a, 1)
            u0 = rng.uniform(0, 0.7)
            u1 = rng.uniform(u0, 1)
            slab = prepare_ridge_slab(ctx, fixed, (a, b))
            side = rng.choice((-1, 1))
            cell = ridge_interval_cell(ctx, slab, side, (u0, u1))
            mu = rng.uniform(a, b)
            u = rng.uniform(u0, u1)
            geometry, eta, jacobian = _point(gamma, B, slab.h, mu, side, u)
            for interval, point in zip(cell.geometry, geometry, strict=True):
                _contains(interval, point)
            _contains(cell.eta, eta)
            _contains(cell.jacobian, jacobian)


def test_both_branches_cover_eta_domain_and_jacobian_integrates_to_two():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 3000.0, 5e-6)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    for mu in (0.1, 0.5, 0.95):
        slab = prepare_ridge_slab(ctx, fixed, (mu, mu))
        total = 0.0
        for side in (-1, 1):
            at_zero = ridge_interval_cell(ctx, slab, side, (0.0, 0.0))
            at_one = ridge_interval_cell(ctx, slab, side, (1.0, 1.0))
            _contains(at_zero.eta, mp.mpf(mu))
            _contains(at_one.eta, mp.mpf(side))
            branch = 0.0
            for node, weight in zip(nodes, weights, strict=True):
                u = float((node + 1) / 2)
                jacobian = ridge_interval_cell(ctx, slab, side, (u, u)).jacobian
                branch += float(weight / 2) * float(jacobian.mid)
            assert math.isclose(branch, 1 - side * mu, rel_tol=1e-12, abs_tol=1e-12)
            total += branch
        assert math.isclose(total, 2.0, rel_tol=1e-12, abs_tol=1e-12)


def test_correlated_cell_tightens_ridge_z_relative_to_independent_angle_box():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 3000.0, 5e-6)
    slab = prepare_ridge_slab(ctx, fixed, (0.49995, 0.50005))
    correlated = ridge_interval_cell(ctx, slab, 1, (0.0, 0.1))
    eta_lo = math.nextafter(float(correlated.eta.a), -math.inf)
    eta_hi = math.nextafter(float(correlated.eta.b), math.inf)
    independent = angular_box_geometry(
        ctx, fixed, (slab.mu_lo, slab.mu_hi), (max(-1, eta_lo), min(1, eta_hi))
    )
    correlated_width = float((correlated.geometry[3].b - correlated.geometry[3].a).mid)
    independent_width = float((independent[3].b - independent[3].a).mid)
    assert correlated_width < independent_width


def test_axis_cells_and_validation_fail_closed():
    ctx = _context()
    fixed = prepare_angular_geometry(ctx, 20.0, 1.0)
    slab = prepare_ridge_slab(ctx, fixed, (0.99, 1.0))
    axis = ridge_interval_cell(ctx, slab, 1, (0.9, 1.0))
    assert axis.eta.b == 1
    assert axis.jacobian.a == 0
    assert axis.geometry[3].a == 0
    assert axis.geometry[0].a > 0
    with pytest.raises(ValueError, match="positive"):
        prepare_ridge_slab(ctx, fixed, (0.0, 0.5), h=0)
    with pytest.raises(ValueError, match="positive-mu slab|ridge slab"):
        prepare_ridge_slab(ctx, fixed, (-0.1, 0.5))
    for side, u_box in ((0, (0.0, 0.1)), (1, (-0.1, 0.1)), (1, (0.5, 0.4))):
        with pytest.raises(ValueError):
            ridge_interval_cell(ctx, slab, side, u_box)
    other = _context()
    with pytest.raises(ValueError, match="interval context"):
        ridge_interval_cell(other, slab, 1, (0.0, 0.1))


def test_high_gamma_near_axis_cell_keeps_positive_D_and_point_containment():
    ctx = _context()
    gamma, B = 1e9, 1e-8
    mu_lo = math.nextafter(1.0, 0.0)
    fixed = prepare_angular_geometry(ctx, gamma, B)
    slab = prepare_ridge_slab(ctx, fixed, (mu_lo, 1.0))
    cell = ridge_interval_cell(ctx, slab, 1, (0.0, 0.1))
    assert cell.geometry[0].a > 0
    assert cell.geometry[-1].a > 0
    for mu, u in ((mu_lo, 0.0), (mu_lo, 0.05), (1.0, 0.1)):
        geometry, eta, jacobian = _point(gamma, B, slab.h, mu, 1, u)
        for interval, point in zip(cell.geometry, geometry, strict=True):
            _contains(interval, point)
        _contains(cell.eta, eta)
        _contains(cell.jacobian, jacobian)


def test_fixed_mu_maximum_tightens_broad_axis_slabs():
    ctx = _context()
    gamma, B = 3000.0, 5e-6
    fixed = prepare_angular_geometry(ctx, gamma, B)
    for mu_box in ((0.0, 1.0), (0.9, 1.0), (1.0, 1.0)):
        slab = prepare_ridge_slab(ctx, fixed, mu_box)
        for side in (-1, 1):
            cell = ridge_interval_cell(ctx, slab, side, (0.0, 1.0))
            assert cell.geometry[3].b < 1
            for mu in (mu_box[0], mu_box[1]):
                for u in (0.0, 0.5, 1.0):
                    point, _, _ = _point(gamma, B, slab.h, mu, side, u)
                    _contains(cell.geometry[3], point[3])


def test_correlated_positive_defect_lower_bound_survives_broad_upper():
    ctx = _context()
    gamma, B = 3000.0, 5e-6
    fixed = prepare_angular_geometry(ctx, gamma, B)
    slab = prepare_ridge_slab(ctx, fixed, (0.998046875, 0.9990234375))
    u_box = (0.8544921875, 0.85546875)
    cell = ridge_interval_cell(ctx, slab, 1, u_box)
    D, delta, _, z, _, _ = cell.geometry
    sine = ctx.sqrt((1 - cell.eta) * (1 + cell.eta))
    defect = (delta / D) ** 2 + (fixed.inverse_gamma * sine / D) ** 2
    assert defect.a > 0 and defect.b > 1
    assert z.b < ctx.mpf("0.96")
    for mu in (slab.mu_lo, slab.mu_hi):
        for u in u_box:
            point, _, _ = _point(gamma, B, slab.h, mu, 1, u)
            _contains(z, point[3])
