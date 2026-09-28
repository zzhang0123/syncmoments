"""Selected harmonic ranges and support pruning before Bessel evaluation."""

import dataclasses

import jax
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncmoments.model import _harmonic_taylor
from syncmoments.model.basis import build_basis
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import Truncation
from syncmoments.model.moments import PopulationSamples, Reference, Support

SUPPORT = Support((1.05, 1.15), (0.99, 1.01), (0.0, 0.0))
REFERENCE = Reference(1.1, 1.0)
CHANNELS = Channels.bump([1e7, 1e8], [1e6, 2e6])


def _kernel(route="product", *, mode_intervals=None):
    return HarmonicKernel(
        100,
        mode_intervals=mode_intervals,
        quadrature=route,
        n_outer=8,
        n_inner=8,
        n_mu=8,
        n_eta=8,
        m_chunk=7,
    )


def test_disjoint_channel_support_prunes_actual_harmonic_input(monkeypatch):
    full = _kernel()
    selected = full.for_support(SUPPORT, CHANNELS, reference=REFERENCE)
    assert selected.selected_intervals() == ((1, 9), (17, 70))
    assert selected.channel_mode_intervals == (((1, 9),), ((17, 70),))
    assert 10 not in np.asarray(selected.harmonics())
    assert selected.harmonics().size == 63 < full.harmonics().size
    assert selected.selected_mode_count() == 63
    assert selected.missing_intervals(SUPPORT, CHANNELS, reference=REFERENCE) == ()

    calls = []
    original = _harmonic_taylor.sum_harmonics

    def counted(fn, ms, chunk):
        calls.append(ms.shape[0])
        return original(fn, ms, chunk)

    monkeypatch.setattr(_harmonic_taylor, "sum_harmonics", counted)
    selected.angular_projection(
        CHANNELS, REFERENCE.gamma0, REFERENCE.B0, truncation=Truncation(0, 0, 0)
    )
    assert calls == [9, 54]


def test_selected_point_modes_equal_full_sum_and_jit():
    full = _kernel()
    selected = full.for_support(SUPPORT, CHANNELS)

    def evaluate(kernel, gamma):
        return kernel.channel_modes(CHANNELS, gamma, 1.0, 0.25, -0.5)

    complete, clipped = evaluate(full, 1.1), evaluate(selected, 1.1)
    for name in ("I", "V", "P"):
        assert_allclose(
            getattr(clipped, name), getattr(complete, name), rtol=1e-10, atol=0
        )
    jitted = jax.jit(lambda gamma: evaluate(selected, gamma))
    assert_allclose(jitted(1.1).I, clipped.I, rtol=1e-11, atol=0)
    full_derivative = jax.grad(lambda g: np.float64(1.0) * evaluate(full, g).I.sum())(
        1.1
    )
    clipped_derivative = jax.grad(lambda g: evaluate(selected, g).I.sum())(1.1)
    assert_allclose(clipped_derivative, full_derivative, rtol=1e-10, atol=0)


@pytest.mark.parametrize("route", ["product", "tensor"])
def test_basis_and_angular_routes_use_selected_modes(route):
    full = _kernel(route)
    selected = full.for_support(SUPPORT, CHANNELS, reference=REFERENCE)
    truncation = Truncation(0, 0, 0)
    direct = full.angular_projection(
        CHANNELS, REFERENCE.gamma0, REFERENCE.B0, truncation=truncation
    )
    clipped = selected.angular_projection(
        CHANNELS, REFERENCE.gamma0, REFERENCE.B0, truncation=truncation
    )
    for name in ("I", "V", "P"):
        assert_allclose(
            getattr(clipped, name), getattr(direct, name), rtol=1e-10, atol=0
        )
    basis = build_basis(
        full, CHANNELS, truncation, REFERENCE, support=SUPPORT, convergence=False
    )
    assert dict(basis.provenance.numerics)["computed_mode_intervals"] == (
        (1, 9),
        (17, 70),
    )
    assert_allclose(basis.I_basis, direct.I, rtol=1e-10, atol=0)
    assert basis.kernel_terms.harmonic_truncation.kind == "bound"


@pytest.mark.parametrize("route", ["product", "tensor"])
def test_selected_first_parameter_derivatives_equal_full_sum(route):
    full = _kernel(route)
    selected = full.for_support(SUPPORT, CHANNELS, reference=REFERENCE)
    kwargs = dict(scales=(0.01, 0.01), truncation=Truncation(0, 0, 1), order=1)
    complete = full.angular_taylor(CHANNELS, 1.1, 1.0, **kwargs)
    clipped = selected.angular_taylor(CHANNELS, 1.1, 1.0, **kwargs)
    for alpha in complete:
        for name in ("I", "V", "P"):
            assert_allclose(
                getattr(clipped[alpha], name),
                getattr(complete[alpha], name),
                rtol=1e-9,
                atol=0,
            )


def test_manual_omission_is_rejected_or_unbounded():
    partial = _kernel(mode_intervals=((17, 25), (40, 100)))
    assert partial.selected_intervals() == ((17, 25), (40, 100))
    assert partial.missing_intervals(SUPPORT, CHANNELS)
    with pytest.raises(ValueError, match="mode_intervals omit"):
        build_basis(
            partial,
            CHANNELS,
            Truncation(0, 0, 0),
            REFERENCE,
            support=SUPPORT,
            convergence=False,
        )
    basis = build_basis(
        partial,
        CHANNELS,
        Truncation(0, 0, 0),
        REFERENCE,
        support=SUPPORT,
        convergence=False,
        allow_truncated=True,
    )
    assert basis.kernel_terms.harmonic_truncation.kind == "unbounded"
    assert "mode_intervals" in basis.kernel_terms.harmonic_truncation.note


def test_channel_selection_is_support_derived_and_global_omissions_are_checked():
    with pytest.raises(TypeError, match="channel_mode_intervals"):
        HarmonicKernel(
            100,
            mode_intervals=((1, 9),),
            channel_mode_intervals=(((1, 9),), ((17, 70),)),
        )
    partial = _kernel(mode_intervals=((1, 9),))
    assert partial.missing_intervals(SUPPORT, CHANNELS)
    with pytest.raises(ValueError, match="mode_intervals omit"):
        build_basis(
            partial,
            CHANNELS,
            Truncation(0, 0, 0),
            REFERENCE,
            support=SUPPORT,
            convergence=False,
        )


def test_pruned_zero_bound_requires_high_B_support_membership():
    selected = _kernel().for_support(SUPPORT, CHANNELS)
    inside = PopulationSamples([1.1], [1.0], [0.2], [0.3], [0.0], [0.0])
    outside = PopulationSamples([1.1], [5.0], [0.2], [0.3], [0.0], [0.0])
    assert selected.truncation_error(SUPPORT, CHANNELS, samples=inside).kind == "bound"
    term = selected.truncation_error(SUPPORT, CHANNELS, samples=outside)
    assert term.kind == "unbounded" and "selected low/gap" in term.note


def test_upper_tail_probe_cannot_hide_out_of_support_low_modes():
    full = _kernel()
    selected = HarmonicKernel(100, tail_probe=10).for_support(SUPPORT, CHANNELS)
    sample = PopulationSamples([1.1], [1.75], [0.9], [0.9], [0.0], [0.0])
    complete = full.channel_modes(CHANNELS, 1.1, 1.75, 0.9, 0.9)
    with pytest.raises(Exception, match="original gamma/B support"):
        selected.channel_modes(CHANNELS, 1.1, 1.75, 0.9, 0.9)
    unchecked = HarmonicKernel(100, mode_intervals=selected.selected_intervals())
    clipped = unchecked.channel_modes(CHANNELS, 1.1, 1.75, 0.9, 0.9)
    assert float(np.abs(complete.I[1] - clipped.I[1])) > 0
    term = selected.truncation_error(SUPPORT, CHANNELS, samples=sample)
    assert term.kind == "unbounded" and "upper-tail probe" in term.note


def test_depth_outside_does_not_invalidate_harmonic_range_bound():
    selected = _kernel().for_support(SUPPORT, CHANNELS)
    sample = PopulationSamples([1.1], [1.0], [0.2], [0.3], [0.0], [1.0])
    assert selected.truncation_error(SUPPORT, CHANNELS, samples=sample).kind == "bound"


def test_product_skips_a_channel_with_no_selected_mode(monkeypatch):
    channels = Channels.bump([1e8, 1e9], [2e6, 1e6])
    selected = _kernel().for_support(SUPPORT, channels)
    assert selected.channel_mode_intervals == (((17, 70),), ())
    calls = []
    original = _harmonic_taylor.sum_harmonics

    def counted(fn, ms, chunk):
        calls.append(ms.shape[0])
        return original(fn, ms, chunk)

    monkeypatch.setattr(_harmonic_taylor, "sum_harmonics", counted)
    projected = selected.angular_projection(
        channels, 1.1, 1.0, truncation=Truncation(0, 0, 0)
    )
    assert calls == [54]
    assert np.all(np.asarray(projected.I[1]) == 0)


def test_pruned_kernel_rejects_reordered_channel_supports():
    selected = _kernel().for_support(SUPPORT, CHANNELS)
    swapped = Channels.bump([1e8, 1e7], [2e6, 1e6])
    with pytest.raises(Exception, match="tied to its original channel supports"):
        selected.angular_projection(swapped, 1.1, 1.0, truncation=Truncation(0, 0, 0))
    with pytest.raises(Exception, match="tied to its original channel supports"):
        selected.channel_modes(swapped, 1.1, 1.0, 0.2, 0.3)
    with pytest.raises(ValueError, match="tied to its original channel supports"):
        build_basis(
            selected,
            swapped,
            Truncation(0, 0, 0),
            REFERENCE,
            support=SUPPORT,
            convergence=False,
        )
    with pytest.raises(TypeError, match="channel_mode_intervals"):
        dataclasses.replace(selected, m_chunk=8)
    assert (
        selected.for_tangents(2).selected_channel_support
        == selected.selected_channel_support
    )


def test_pruned_kernel_rejects_parameter_points_and_widened_support():
    selected = _kernel().for_support(SUPPORT, CHANNELS)
    assert selected.selected_parameter_support == (SUPPORT.gamma, SUPPORT.B)
    with pytest.raises(Exception, match="original gamma/B support"):
        selected.channel_modes(CHANNELS, 1.1, 1.75, 0.2, 0.3)
    with pytest.raises(Exception, match="original gamma/B support"):
        selected.angular_projection(CHANNELS, 1.2, 1.0, truncation=Truncation(0, 0, 0))
    jitted = jax.jit(lambda B: selected.channel_modes(CHANNELS, 1.1, B, 0.2, 0.3).I)
    with pytest.raises(Exception, match="original gamma/B support"):
        jitted(1.75).block_until_ready()
    wider = Support(SUPPORT.gamma, (0.99, 1.75), SUPPORT.depth)
    with pytest.raises(ValueError, match="cannot be widened"):
        selected.for_support(wider, CHANNELS)


def test_empty_selected_range_has_zero_outputs_and_no_bessel_call(monkeypatch):
    distant = Channels.bump([1e9], [1e6])
    selected = _kernel().for_support(SUPPORT, distant)
    assert selected.selected_intervals() == ()

    def forbidden(*args, **kwargs):
        raise AssertionError("Bessel evaluator called for an empty range")

    monkeypatch.setattr(_harmonic_taylor, "harmonic_lines", forbidden)
    modes = selected.channel_modes(distant, 1.1, 1.0, 0.2, 0.3)
    assert np.all(np.asarray(modes.I) == 0)
    basis = build_basis(
        selected,
        distant,
        Truncation(0, 0, 0),
        REFERENCE,
        support=SUPPORT,
        convergence=False,
        allow_truncated=True,
    )
    assert np.all(np.asarray(basis.I_basis) == 0)
    assert dict(basis.provenance.numerics)["computed_mode_intervals"] == ()
    assert basis.kernel_terms.harmonic_truncation.kind == "unbounded"

    changed = Channels.bump([2e9], [1e6])
    jitted = jax.jit(lambda ch: selected.channel_modes(ch, 1.1, 1.0, 0.2, 0.3).I)
    with pytest.raises(Exception, match="tied to its original channel supports"):
        jitted(changed).block_until_ready()
    jitted_projection = jax.jit(
        lambda ch: selected.angular_projection(
            ch, 1.1, 1.0, truncation=Truncation(0, 0, 0)
        ).P
    )
    with pytest.raises(Exception, match="tied to its original channel supports"):
        jitted_projection(changed).block_until_ready()


@pytest.mark.parametrize(
    "intervals",
    [((0, 5),), ((5, 101),), ((5.0, 10),), ((10, 5),), (5, 10)],
)
def test_invalid_manual_intervals_rejected(intervals):
    with pytest.raises(ValueError, match="mode_intervals"):
        _kernel(mode_intervals=intervals)
