"""Finite checks of the eager low Taylor-order high-harmonic moment basis."""

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
    JointMoments,
    Reference,
    Support,
    TaylorPhase,
    Truncation,
    predict,
)
from syncmoments.model.high_order_basis import build_high_order_basis


def _setup():
    gamma, B = 20.0, 1.0
    nu_b = E_ESU * B / (2 * np.pi * gamma * M_E * C_CGS)
    channels = Channels.bump([5 * nu_b], [2 * nu_b])
    reference = Reference(gamma, B, depth_ref=0.1, scales=(gamma, B, 0.2))
    support = Support((19.0, 21.0), (0.9, 1.1), (-1.0, 1.0), truncated=False)
    truncation = Truncation(1, 1, 0, depth_degree=1)
    return channels, reference, support, truncation


def test_n0_basis_matches_existing_full_tensor_projection_and_parity():
    channels, reference, support, truncation = _setup()
    phase = TaylorPhase(1)
    basis = build_high_order_basis(
        channels,
        truncation,
        reference,
        support=support,
        phase=phase,
        n_mu=8,
        n_eta=8,
        angular_rule="tensor",
    )
    old = HarmonicKernel(20, n_mu=8, n_eta=8, quadrature="tensor")
    projected = old.angular_projection(
        channels,
        reference.gamma0,
        reference.B0,
        phase=phase,
        depth_ref=reference.depth_ref,
        s_depth=reference.scales[2],
        truncation=basis.index,
        quadrature="tensor",
    )

    pair_position = {pair: pos for pos, pair in enumerate(basis.index.pairs)}
    for name, rows, parity in (
        ("I", basis.index.h0, basis.index.parity_I()),
        ("V", basis.index.h0, basis.index.parity_V()),
    ):
        expected = np.stack(
            [
                np.asarray(getattr(projected, name))[:, pair_position[(l, k)]]
                * parity[pos]
                for pos, (l, k, _, _, _) in enumerate(rows)
            ],
            axis=1,
        )
        actual = np.asarray(getattr(basis, f"{name}_basis"))
        assert_allclose(actual, expected, rtol=1e-10, atol=2e-31)
    expected_P = np.stack(
        [
            np.asarray(projected.P)[b, :, pair_position[(l, k)]]
            for l, k, _, _, b in basis.index.h2
        ],
        axis=1,
    )
    assert_allclose(np.asarray(basis.P_basis), expected_P, rtol=1e-10, atol=2e-31)
    assert np.any(np.asarray(basis.V_basis) != 0)
    assert basis.kernel_terms.numerical.kind == "unbounded"
    assert basis.kernel_terms.harmonic_truncation.kind == "bound"
    assert basis.certified_orders == ()
    assert dict(basis.provenance.numerics)["angular_evaluations"] == 32


def test_prediction_reuses_prepared_matrix_without_harmonic_calls(monkeypatch):
    channels, reference, support, truncation = _setup()
    basis = build_high_order_basis(
        channels, truncation, reference, support=support, n_mu=4, n_eta=4
    )
    vector = np.zeros(basis.index.n_real)
    vector[0] = 1.0
    vector[1:] = 0.1
    moments = JointMoments.from_vector(basis.index, vector, reference)
    expected = (np.asarray(basis.response_matrix()) @ vector).reshape(1, 4)

    def fail(*args, **kwargs):
        raise AssertionError("predict must reuse the prepared matrix")

    monkeypatch.setattr(
        "syncmoments.model.high_order_basis.high_order_channel_modes", fail
    )
    for amplitude in (1.0, 2.0):
        result = predict(basis, moments, amplitude=amplitude)
        assert_allclose(np.asarray(result.stokes), amplitude * expected, rtol=1e-13)
        assert result.budget.numerical.kind == "unbounded"

    def forward(m):
        joint = JointMoments.from_vector(basis.index, m, reference)
        return predict(basis, joint, amplitude=1.0).stokes.ravel()

    jitted = jax.jit(forward)
    assert_allclose(np.asarray(jitted(jnp.asarray(vector))), expected.ravel())
    jacobian = np.asarray(jax.jit(jax.jacfwd(forward))(jnp.asarray(vector)))
    assert_allclose(jacobian, np.asarray(basis.response_matrix()), rtol=1e-13)


def test_n1_basis_derivative_columns_match_fixed_grid_finite_differences():
    channels, reference, support, _ = _setup()
    # The low-order test channel is near a MHz: depth_ref=0.1 would rotate
    # through many radians over an otherwise useful finite-difference step.
    reference = Reference(
        reference.gamma0,
        reference.B0,
        depth_ref=1.0e-6,
        scales=reference.scales,
    )
    settings = dict(
        support=support,
        phase=TaylorPhase(1),
        n_mu=8,
        n_eta=8,
        angular_rule="tensor",
    )
    truncation = Truncation(1, 1, 1, depth_degree=1)
    basis = build_high_order_basis(channels, truncation, reference, **settings)
    zeroth = Truncation(1, 1, 0, depth_degree=1)
    epsilon = 1.0e-5
    for derivative in ("gamma", "B"):
        d_gamma = reference.scales[0] * epsilon if derivative == "gamma" else 0.0
        d_B = reference.scales[1] * epsilon if derivative == "B" else 0.0
        displaced = []
        for sign in (+1.0, -1.0):
            shifted = Reference(
                reference.gamma0 + sign * d_gamma,
                reference.B0 + sign * d_B,
                depth_ref=reference.depth_ref,
                scales=reference.scales,
            )
            displaced.append(
                build_high_order_basis(channels, zeroth, shifted, **settings)
            )
        r, s = (1, 0) if derivative == "gamma" else (0, 1)
        for component, rows in (
            ("I", basis.index.h0),
            ("V", basis.index.h0),
            ("P", basis.index.h2),
        ):
            value = np.asarray(getattr(basis, f"{component}_basis"))
            high = np.asarray(getattr(displaced[0], f"{component}_basis"))
            low = np.asarray(getattr(displaced[1], f"{component}_basis"))
            old_rows = (
                displaced[0].index.h0 if component != "P" else displaced[0].index.h2
            )
            for pos, (l, k, row_r, row_s, b) in enumerate(rows):
                if (row_r, row_s) != (r, s):
                    continue
                old_pos = old_rows.index((l, k, 0, 0, b))
                expected = (high[:, old_pos] - low[:, old_pos]) / (2.0 * epsilon)
                assert_allclose(value[:, pos], expected, rtol=2e-5, atol=2e-27)
    assert basis.kernel_terms.numerical.kind == "unbounded"
    assert basis.certified_orders == ()


def test_builder_refuses_unavailable_derivatives_and_invalid_rule():
    channels, reference, support, _ = _setup()
    with pytest.raises(ValueError, match="N <= 1"):
        build_high_order_basis(
            channels, Truncation(1, 1, 2), reference, support=support
        )
    with pytest.raises(ValueError, match="require bump"):
        build_high_order_basis(
            Channels.planck_taper(
                [float(channels.centres_hz[0])], [float(channels.widths_hz[0])]
            ),
            Truncation(1, 1, 1),
            reference,
            support=support,
        )
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
        build_high_order_basis(
            clipped,
            Truncation(0, 0, 1),
            reference,
            support=support,
            n_mu=4,
            n_eta=4,
        )
    with pytest.raises(ValueError, match="even"):
        build_high_order_basis(
            channels, Truncation(1, 1, 0), reference, support=support, n_mu=3
        )
    with pytest.raises(ValueError, match="angular_rule"):
        build_high_order_basis(
            channels,
            Truncation(1, 1, 0),
            reference,
            support=support,
            angular_rule="unknown",
        )
    with pytest.raises(ValueError, match="max_b"):
        build_high_order_basis(
            channels,
            Truncation(1, 1, 0, depth_degree=1),
            reference,
            support=support,
            phase=TaylorPhase(0),
        )


def test_galactic_broad_channels_prepare_finite_matrix_without_mode_array():
    channels = Channels.bump([1e8, 1e9, 3e9], [2e7, 2e8, 6e8])
    reference = Reference(3000.0, 5e-6, scales=(3000.0, 5e-6, 1.0))
    support = Support((2000.0, 4000.0), (4e-6, 6e-6), (-1.0, 1.0))
    basis = build_high_order_basis(
        channels,
        Truncation(1, 1, 0),
        reference,
        support=support,
        n_mu=4,
        n_eta=4,
        allow_unconverged=True,
    )
    numerics = dict(basis.provenance.numerics)
    assert numerics["active_order_envelope"][1] > 10**11
    assert numerics["real_order_samples"] < 100_000
    assert (
        sum(count for _, count in numerics["routes"]) == numerics["angular_evaluations"]
    )
    assert dict(basis.provenance.quadrature)["route"] == "ridge_positive_mu_parity"
    assert np.all(np.isfinite(np.asarray(basis.response_matrix())))
    assert basis.kernel_terms.numerical.kind == "unbounded"


def test_galactic_n1_basis_prepares_scaled_derivatives_without_mode_array(monkeypatch):
    channels = Channels.bump([1e8], [2e7])
    reference = Reference(3000.0, 5e-6, depth_ref=0.1, scales=(3000.0, 5e-6, 1.0))
    support = Support((2000.0, 4000.0), (4e-6, 6e-6), (-1.0, 1.0))
    basis = build_high_order_basis(
        channels,
        Truncation(0, 0, 1, depth_degree=0),
        reference,
        support=support,
        n_mu=8,
        n_eta=12,
        atol=1e-30,
        atol_gamma=1e-27,
        atol_B=1e-25,
        rtol=1e-5,
    )
    numerics = dict(basis.provenance.numerics)
    assert numerics["active_order_envelope"][1] > 10**10
    assert numerics["real_order_samples"] < 100_000
    assert numerics["unconverged_node_channels"] == 0
    assert np.all(np.isfinite(np.asarray(basis.response_matrix())))
    assert basis.kernel_terms.numerical.kind == "unbounded"
    vector = jnp.asarray([1.0] + [0.1] * (basis.index.n_real - 1))
    expected = np.asarray(basis.response_matrix() @ vector).reshape(1, 4)

    def fail(*args, **kwargs):
        raise AssertionError("prepared N=1 prediction must not evaluate harmonics")

    monkeypatch.setattr(
        "syncmoments.model.high_order_basis.high_order_channel_derivatives", fail
    )

    @jax.jit
    def spectra(moment_vector):
        moments = JointMoments.from_vector(basis.index, moment_vector, reference)
        return predict(basis, moments, amplitude=1.0).stokes

    assert_allclose(np.asarray(spectra(vector)), expected, rtol=1e-13)


def test_galactic_ridge_projection_has_finite_refinement_agreement():
    """A tensor rule can alias the narrow eta=mu ridge at gamma=3000.

    The 1e-30 per-electron Stokes atol is solely an inner quadrature stopping
    check in erg/s/sr for this unit-peak channel, not a total error bound.
    """
    channels = Channels.bump([1e8], [2e7])
    reference = Reference(3000.0, 5e-6, scales=(3000.0, 5e-6, 1.0))
    support = Support((2000.0, 4000.0), (4e-6, 6e-6), (-1.0, 1.0))
    values = []
    for n_mu, n_eta in ((8, 24), (16, 24), (24, 24), (24, 25), (24, 32), (32, 32)):
        basis = build_high_order_basis(
            channels,
            Truncation(0, 0, 0),
            reference,
            support=support,
            n_mu=n_mu,
            n_eta=n_eta,
            atol=1e-30,
        )
        assert basis.kernel_terms.numerical.kind == "unbounded"
        assert dict(basis.provenance.numerics)["unconverged_node_channels"] == 0
        values.append(float(np.asarray(basis.I_basis)[0, 0]))
    assert np.all(np.asarray(values) > 0)
    assert_allclose(values[:4], values[4], rtol=1e-3)
    assert_allclose(values[4], values[5], rtol=1e-5)


def test_widely_separated_channels_use_distinct_ridge_scales():
    """A low-frequency eta scale undersamples a distant high channel."""
    channels = Channels.bump([1e8, 3e10], [2e7, 6e9])
    reference = Reference(30000.0, 5e-6, scales=(30000.0, 5e-6, 1.0))
    support = Support((20000.0, 40000.0), (4e-6, 6e-6), (-1.0, 1.0))
    values = []
    for n_eta in (24, 48):
        basis = build_high_order_basis(
            channels,
            Truncation(0, 0, 0),
            reference,
            support=support,
            n_mu=8,
            n_eta=n_eta,
            atol=1e-28,
            allow_unconverged=True,
        )
        assert basis.kernel_terms.numerical.kind == "unbounded"
        values.append(np.asarray(basis.I_basis)[:, 0])
    assert np.all(np.asarray(values) > 0)
    assert_allclose(values[0], values[1], rtol=1e-2)
    high_only = build_high_order_basis(
        Channels.bump([3e10], [6e9]),
        Truncation(0, 0, 0),
        reference,
        support=support,
        n_mu=8,
        n_eta=24,
        atol=1e-28,
        allow_unconverged=True,
    )
    assert_allclose(values[0][1], np.asarray(high_only.I_basis)[0, 0], rtol=1e-12)
