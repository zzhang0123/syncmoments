"""Finite checks of the optional conditional coarse N<=1 matrix certificate."""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from syncmoments.constants import C_SI_M
from syncmoments.model import (
    Channels,
    ErrorTerm,
    JointMoments,
    Reference,
    SpectralBasis,
    Support,
    Truncation,
    build_high_order_basis,
    certify_high_order_basis_coarse,
    predict,
)
from syncmoments.model._certified_angular_i00 import AngularI00Limit
from syncmoments.model._certified_high_order_basis import CertifiedHighOrderBasis


@jax.jit
def _jitted_matrix(basis):
    return basis.response_matrix()


@jax.jit
def _jitted_stokes(basis, moments):
    return predict(basis, moments, amplitude=1.0).stokes


@jax.jit
def _jitted_validated_numerical(basis):
    return basis.validated_numerical_term().value


@jax.jit
def _jitted_predict_budget_only(basis, moments):
    return predict(basis, moments, amplitude=1.0).budget.numerical.value


def _low_basis(*, n_mu=8, n_eta=8):
    channels = Channels.bump([1e7], [1e6])
    reference = Reference(2.0, 1.0, scales=(2.0, 1.0, 0.2))
    support = Support((1.5, 2.5), (0.9, 1.1), (-1.0, 1.0), truncated=False)
    return build_high_order_basis(
        channels,
        Truncation(1, 1, 0, depth_degree=1),
        reference,
        support=support,
        n_mu=n_mu,
        n_eta=n_eta,
        angular_rule="tensor",
        allow_unconverged=True,
    )


def _certify(basis, **kwargs):
    return certify_high_order_basis_coarse(
        basis,
        i00_atol=1e-12,
        max_cells=64,
        max_evaluated_cells=128,
        max_mode_blocks=0,
        **kwargs,
    )


def _with_arrays(basis, *, I_basis=None, V_basis=None, P_basis=None, provenance=None):
    return SpectralBasis(
        index=basis.index,
        truncation=basis.truncation,
        channels=basis.channels,
        reference=basis.reference,
        support=basis.support,
        I_basis=basis.I_basis if I_basis is None else I_basis,
        V_basis=basis.V_basis if V_basis is None else V_basis,
        P_basis=basis.P_basis if P_basis is None else P_basis,
        kernel_terms=basis.kernel_terms,
        provenance=basis.provenance if provenance is None else provenance,
        certified_orders=basis.certified_orders,
        phase=basis.phase,
    )


def test_low_order_matrix_certificate_preserves_fast_matrix_and_exact_zeros():
    basis = _low_basis()
    original = np.asarray(basis.response_matrix()).copy()
    assert basis.kernel_terms.numerical.kind == "unbounded"
    with pytest.raises(ValueError, match="valued numerical bound"):
        CertifiedHighOrderBasis(basis)
    certified, report = _certify(basis)
    assert isinstance(certified, CertifiedHighOrderBasis)
    matrix = np.asarray(certified.response_matrix())
    assert np.array_equal(np.asarray(_jitted_matrix(certified)), original)
    assert np.array_equal(
        np.asarray(_jitted_validated_numerical(certified)),
        np.asarray(certified.kernel_terms.numerical.value),
    )
    errors = np.asarray(certified.kernel_terms.numerical.value)
    assert np.array_equal(matrix, original)
    assert basis.kernel_terms.numerical.kind == "unbounded"
    assert certified.kernel_terms.numerical.kind == "bound"
    assert errors.shape == matrix.shape
    assert report.matrix_shape == matrix.shape
    assert report.matrix_dtype == str(matrix.dtype)
    assert len(report.matrix_sha256) == 64
    assert len(report.channel_certificates) == 1
    assert len(report.maximum_column_error) == basis.index.n_real
    assert "experimental mpmath" in certified.kernel_terms.numerical.note
    assert "certified_matrix" in dict(certified.provenance.numerics)

    for entry in basis.index.entries():
        if entry.h == 0:
            allowed = 0 if (entry.l + entry.k) % 2 == 0 else 3
            assert np.all(
                errors[[i for i in range(4) if i != allowed], entry.slot] == 0
            )
        else:
            assert errors[0, entry.slot] == errors[3, entry.slot] == 0

    # A second finite angular rule is a diagnostic only. It is contained by
    # this deliberately loose envelope, but does not prove the interval work.
    other = np.asarray(_low_basis(n_mu=12, n_eta=12).response_matrix())
    assert np.all(np.abs(original - other) <= errors)


def test_taylor_depth_weight_and_predict_numerical_budget():
    basis = _low_basis()
    certified, report = _certify(basis)
    errors = np.asarray(certified.kernel_terms.numerical.value)
    matrix = np.asarray(certified.response_matrix())
    interval = report.channel_certificates[0].interval
    upper = float(interval.b)
    frequency_lo = float(np.asarray(basis.channels.support)[0, 0])
    s_depth = float(np.asarray(basis.reference.scales[2]))
    weight_1 = 2 * (C_SI_M / frequency_lo) ** 2 * s_depth
    for entry in basis.index.entries():
        if entry.h != 2:
            continue
        bound = (2 * entry.l + 1) * (2 * entry.k + 1) * upper
        if entry.b == 1:
            bound *= weight_1
        for stokes in (1, 2):
            # Error includes the stored binary coefficient and both sides of
            # its symmetric exact-coefficient interval.
            assert errors[stokes, entry.slot] >= abs(
                matrix[stokes, entry.slot]
            ) + bound * (1 - 1e-13)

    vector = np.zeros(basis.index.n_real)
    vector[0] = 1
    vector[-1] = 0.25
    moments = JointMoments.from_vector(basis.index, vector, basis.reference)
    predicted = predict(certified, moments, amplitude=2.0)
    expected = 2 * (errors @ np.abs(vector)).reshape(1, 4)
    assert predicted.budget.numerical.kind == "bound"
    np.testing.assert_allclose(np.asarray(predicted.budget.numerical.value), expected)


def test_float32_publication_is_outward_in_stored_dtype():
    basis = _low_basis(n_mu=4, n_eta=4)
    float32_basis = _with_arrays(
        basis,
        I_basis=jnp.asarray(basis.I_basis, dtype=jnp.float32),
        V_basis=jnp.asarray(basis.V_basis, dtype=jnp.float32),
        P_basis=jnp.asarray(basis.P_basis, dtype=jnp.complex64),
    )
    matrix = np.asarray(float32_basis.response_matrix())
    assert matrix.dtype == np.dtype("float32")
    certified, report = _certify(float32_basis)
    published = np.asarray(certified.kernel_terms.numerical.value)
    assert published.dtype == np.dtype("float32")
    assert report.matrix_dtype == "float32"
    with mp.workdps(100):
        interval = report.channel_certificates[0].interval
        lo = mp.mpf(interval.a._mpi_[0])
        hi = mp.mpf(interval.b._mpi_[1])
        c = mp.mpf(float(matrix[0, 0]))
        required = max(abs(c - lo), abs(hi - c))
        assert mp.mpf(float(published[0, 0])) >= required
    assert np.array_equal(np.asarray(certified.response_matrix()), matrix)


def test_bad_provenance_resource_cap_and_column_budget_fail_closed():
    basis = _low_basis(n_mu=4, n_eta=4)
    changed = dataclasses.replace(basis.provenance, channels=())
    with pytest.raises(ValueError, match="provenance"):
        _certify(_with_arrays(basis, provenance=changed))
    changed_phase = dataclasses.replace(basis.provenance, phase_route=())
    with pytest.raises(ValueError, match="provenance"):
        _certify(_with_arrays(basis, provenance=changed_phase))
    with pytest.raises(AngularI00Limit, match="cell cap"):
        certify_high_order_basis_coarse(
            basis,
            i00_atol=1e-30,
            max_cells=2,
            max_evaluated_cells=4,
            max_mode_blocks=0,
        )
    with pytest.raises(ArithmeticError, match="column_atol unmet"):
        _certify(basis, column_atol=np.zeros(basis.index.n_real))
    assert basis.kernel_terms.numerical.kind == "unbounded"
    with pytest.raises(ValueError, match="column_atol"):
        _certify(basis, column_atol=-1)


def test_each_channel_gets_its_own_i00_interval_and_column_budget():
    channels = Channels.bump([1e7, 2e7], [1e6, 2e6])
    reference = Reference(2.0, 1.0, scales=(2.0, 1.0, 1.0))
    support = Support((1.5, 2.5), (0.9, 1.1), (-1.0, 1.0))
    basis = build_high_order_basis(
        channels,
        Truncation(0, 0, 0),
        reference,
        support=support,
        n_mu=4,
        n_eta=4,
        allow_unconverged=True,
    )
    certified, report = _certify(basis)
    assert len(report.channel_certificates) == 2
    assert np.asarray(certified.kernel_terms.numerical.value).shape == (8, 3)
    assert np.all(np.asarray(certified.kernel_terms.numerical.value)[[3, 7], :] == 0)
    budget = tuple(2 * value for value in report.maximum_column_error)
    again, _ = _certify(basis, column_atol=budget)
    assert np.array_equal(
        np.asarray(again.response_matrix()), np.asarray(basis.response_matrix())
    )


def test_high_gamma_broad_channel_can_attach_offline_coarse_bound():
    channels = Channels.bump([1e8], [1e7])
    reference = Reference(3000.0, 5e-6, scales=(3000.0, 5e-6, 1.0))
    support = Support((2000.0, 4000.0), (4e-6, 6e-6), (-1.0, 1.0))
    basis = build_high_order_basis(
        channels,
        Truncation(0, 0, 0),
        reference,
        support=support,
        n_mu=24,
        n_eta=32,
        atol=1e-30,
    )
    certified, report = certify_high_order_basis_coarse(
        basis,
        i00_atol=1e-18,
        max_cells=512,
        max_evaluated_cells=1024,
        max_mode_blocks=900,
    )
    assert certified.kernel_terms.numerical.kind == "bound"
    assert np.array_equal(
        np.asarray(certified.response_matrix()), np.asarray(basis.response_matrix())
    )
    assert report.channel_certificates[0].absolute_error <= 1e-18
    assert report.channel_certificates[0].mode_blocks_evaluated > 0
    assert math.isfinite(report.elapsed_seconds)


def test_n1_derivative_columns_attach_offline_bound_and_propagate():
    channels = Channels.bump([1e7], [1e6])
    reference = Reference(2.0, 1.0, depth_ref=1e-6, scales=(2.0, 1.0, 0.2))
    support = Support((1.5, 2.5), (0.9, 1.1), (-1.0, 1.0), truncated=False)
    basis = build_high_order_basis(
        channels,
        Truncation(1, 1, 1, depth_degree=1),
        reference,
        support=support,
        n_mu=8,
        n_eta=8,
        angular_rule="tensor",
    )
    original = np.asarray(basis.response_matrix()).copy()
    certified, report = _certify(basis)
    errors = np.asarray(certified.validated_numerical_term().value)
    assert np.array_equal(np.asarray(certified.response_matrix()), original)
    assert report.channel_certificates[0].derivative_upper is not None
    assert dict(report.configuration)["first_derivatives"] is True
    assert np.all(np.isfinite(errors))
    assert np.all(errors >= 0)
    assert np.any(errors[:, basis.index.position(0, 0, 0, 1, 0, 0)] > 0)
    assert np.any(errors[:, basis.index.position(0, 0, 0, 0, 1, 0)] > 0)
    assert np.array_equal(np.asarray(_jitted_matrix(certified)), original)
    vector = np.linspace(0.1, 1.0, basis.index.n_real)
    vector[0] = 1.0
    moments = JointMoments.from_vector(basis.index, vector, reference)
    predicted = predict(certified, moments, amplitude=1.0)
    expected = (errors @ np.abs(vector)).reshape(1, 4)
    np.testing.assert_allclose(np.asarray(predicted.budget.numerical.value), expected)
    assert predicted.budget.basis_remainder.kind == "unbounded"
    with pytest.raises(ArithmeticError, match="column_atol unmet"):
        _certify(basis, column_atol=0.0)


def test_n1_high_gamma_coarse_certificate_keeps_inference_matrix_unchanged():
    channels = Channels.bump([1e8], [2e7])
    reference = Reference(3000.0, 5e-6, scales=(3000.0, 5e-6, 1.0))
    support = Support((2000.0, 4000.0), (4e-6, 6e-6), (-1.0, 1.0))
    basis = build_high_order_basis(
        channels,
        Truncation(0, 0, 1, depth_degree=0),
        reference,
        support=support,
        n_mu=8,
        n_eta=12,
        rtol=1e-5,
        atol=1e-30,
        atol_gamma=1e-27,
        atol_B=1e-25,
    )
    original = np.asarray(basis.response_matrix()).copy()
    certified, report = certify_high_order_basis_coarse(
        basis,
        i00_atol=2e-18,
        max_cells=512,
        max_evaluated_cells=1024,
        max_mode_blocks=900,
    )
    assert np.array_equal(np.asarray(certified.response_matrix()), original)
    assert report.channel_certificates[0].derivative_upper is not None
    assert math.isfinite(report.elapsed_seconds)
    assert all(
        math.isfinite(value) and value > 0 for value in report.maximum_column_error
    )


@pytest.fixture(scope="module")
def _guarded_basis():
    return _certify(_low_basis(n_mu=4, n_eta=4))[0]


@pytest.mark.parametrize(
    "field",
    (
        "coefficient",
        "error_value",
        "error_note",
        "channel_centre",
        "channel_width",
        "channel_support",
        "channel_node",
        "reference_gamma",
        "reference_depth_scale",
        "support_gamma",
        "support_truncated",
    ),
)
def test_changed_certificate_identity_fails_eager_and_jit(_guarded_basis, field):
    basis = _guarded_basis
    if field == "coefficient":
        changed = eqx.tree_at(
            lambda x: x.I_basis, basis, basis.I_basis.at[0, 0].add(1e-15)
        )
    elif field == "error_value":
        error = basis.kernel_terms.numerical.value
        changed = eqx.tree_at(
            lambda x: x.kernel_terms.numerical.value,
            basis,
            error.at[0, 0].set(0),
        )
    elif field == "error_note":
        previous = basis.kernel_terms.numerical
        altered = ErrorTerm(previous.value, "bound", "changed note", "E_num")
        changed = eqx.tree_at(lambda x: x.kernel_terms.numerical, basis, altered)
    elif field == "channel_centre":
        changed = eqx.tree_at(
            lambda x: x.channels.centres_hz,
            basis,
            basis.channels.centres_hz * 1.01,
        )
    elif field == "channel_width":
        changed = eqx.tree_at(
            lambda x: x.channels.widths_hz,
            basis,
            basis.channels.widths_hz * 1.01,
        )
    elif field == "channel_support":
        changed = eqx.tree_at(
            lambda x: x.channels.support,
            basis,
            basis.channels.support.at[0, 0].multiply(1.01),
        )
    elif field == "channel_node":
        changed = eqx.tree_at(
            lambda x: x.channels.nodes,
            basis,
            basis.channels.nodes.at[0, 0].multiply(1.01),
        )
    elif field == "reference_gamma":
        changed = eqx.tree_at(
            lambda x: x.reference.gamma0,
            basis,
            basis.reference.gamma0 + 0.1,
        )
    elif field == "reference_depth_scale":
        changed = eqx.tree_at(
            lambda x: x.reference.scales[2],
            basis,
            basis.reference.scales[2] * 1.01,
        )
    elif field == "support_gamma":
        changed = eqx.tree_at(
            lambda x: x.support.gamma[1],
            basis,
            basis.support.gamma[1] + 0.1,
        )
    else:
        replacement = Support(
            tuple(float(x) for x in basis.support.gamma),
            tuple(float(x) for x in basis.support.B),
            tuple(float(x) for x in basis.support.depth),
            truncated=True,
        )
        changed = eqx.tree_at(lambda x: x.support, basis, replacement)

    with pytest.raises(Exception, match="certified high-order basis changed"):
        np.asarray(changed.response_matrix())
    with pytest.raises(Exception, match="certified high-order basis changed"):
        np.asarray(_jitted_matrix(changed))
    with pytest.raises(Exception, match="certified high-order basis changed"):
        np.asarray(changed.validated_numerical_term().value)
    with pytest.raises(Exception, match="certified high-order basis changed"):
        np.asarray(_jitted_validated_numerical(changed))


@pytest.mark.parametrize("field", ("coefficient", "error_value"))
def test_predict_stokes_refuses_changed_coefficient_or_envelope(_guarded_basis, field):
    basis = _guarded_basis
    if field == "coefficient":
        changed = eqx.tree_at(
            lambda x: x.I_basis, basis, basis.I_basis.at[0, 0].add(1e-15)
        )
    else:
        error = basis.kernel_terms.numerical.value
        changed = eqx.tree_at(
            lambda x: x.kernel_terms.numerical.value,
            basis,
            jnp.zeros_like(error),
        )
    vector = np.zeros(basis.index.n_real)
    vector[0] = 1
    moments = JointMoments.from_vector(basis.index, vector, basis.reference)
    with pytest.raises(Exception, match="certified high-order basis changed"):
        np.asarray(predict(changed, moments, amplitude=1.0).stokes)
    with pytest.raises(Exception, match="certified high-order basis changed"):
        np.asarray(_jitted_stokes(changed, moments))


@pytest.mark.parametrize("field", ("coefficient", "error_value"))
def test_predict_budget_only_refuses_stale_certificate(_guarded_basis, field):
    basis = _guarded_basis
    vector = np.zeros(basis.index.n_real)
    vector[0] = 1
    moments = JointMoments.from_vector(basis.index, vector, basis.reference)
    expected = np.asarray(predict(basis, moments, amplitude=1.0).budget.numerical.value)
    np.testing.assert_array_equal(
        np.asarray(_jitted_predict_budget_only(basis, moments)), expected
    )
    if field == "coefficient":
        changed = eqx.tree_at(
            lambda x: x.I_basis, basis, basis.I_basis.at[0, 0].add(1e-15)
        )
    else:
        error = basis.kernel_terms.numerical.value
        changed = eqx.tree_at(
            lambda x: x.kernel_terms.numerical.value,
            basis,
            jnp.zeros_like(error),
        )
    with pytest.raises(Exception, match="certified high-order basis changed"):
        np.asarray(predict(changed, moments, amplitude=1.0).budget.numerical.value)
    with pytest.raises(Exception, match="certified high-order basis changed"):
        np.asarray(_jitted_predict_budget_only(changed, moments))
