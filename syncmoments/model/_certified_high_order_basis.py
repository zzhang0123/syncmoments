"""Optional, conditional coarse numerical certificate for a high-order N<=1 basis.

One full-domain angular I00 enclosure per channel controls every retained N=0
coefficient. At N=1, whole-angle gamma/B derivative magnitude envelopes also
control the two physical-parameter columns. Positivity of intensity,
|V| <= I, |Q+iU| <= I, and
|P_l(mu) P_k(eta)| <= 1 imply a magnitude at most
``(2*l+1)*(2*k+1)*I00_upper`` for an allowed angular coefficient.  The
TaylorPhase depth coefficient also receives ``|tau*s_depth|**b/b!``, with
``tau <= 2*(c/nu_support_lo)**2``.  Exact parity and Stokes-layout zeros are
handled separately.  This is deliberately coarse: it certifies the cached
matrix only at its fixed binary gamma/B reference and does not bound a
parameter displacement, model discrepancy, population tail or fit error.

The underlying full-domain I00 enclosure is conditional on mpmath's
experimental interval elementary functions, as well as the stated ideal
binary-input unit-peak bump model.  The host SciPy calculation is not assumed
accurate: every actual stored binary matrix entry is compared to the exact
coefficient enclosure.  No certificate calculation runs in predict or fit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from numbers import Real
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..constants import C_SI_M
from ._basis_checks import UNITS, check_phase, truncation_record
from ._certified_angular_i00 import AngularI00Certificate, certify_angular_i00
from .basis import KernelTerms, SpectralBasis
from .errors import ErrorTerm
from .high_order_harmonic import _single_channel
from .index import MomentIndex
from .phase import TaylorPhase


@dataclasses.dataclass(frozen=True)
class CoarseHighOrderBasisReport:
    """Auditable configuration, cached-matrix identity and measured offline work."""

    matrix_shape: tuple[int, int]
    matrix_dtype: str
    matrix_sha256: str
    configuration: tuple[tuple[str, object], ...]
    channel_certificates: tuple[AngularI00Certificate, ...]
    maximum_column_error: tuple[float, ...]
    elapsed_seconds: float


def _dynamic_inputs(basis):
    """Every dynamic input used to identify this fixed-response certificate."""
    channels, reference, support = basis.channels, basis.reference, basis.support
    return (
        channels.centres_hz,
        channels.widths_hz,
        channels.support,
        channels.nodes,
        channels.weights,
        reference.gamma0,
        reference.B0,
        reference.depth_ref,
        *reference.scales,
        *support.gamma,
        *support.B,
        *support.depth,
    )


def _static_identity(basis):
    """The exact static model, layout, certificate label and source record."""
    channels = basis.channels
    numerical = basis.kernel_terms.numerical
    return (
        basis.index,
        basis.truncation,
        basis.phase,
        basis.provenance,
        basis.certified_orders,
        channels.family,
        channels.normalisation,
        channels.smoothness,
        channels.taper,
        channels.n_nu,
        channels.table,
        channels.support_sigma,
        basis.support.truncated,
        numerical.kind,
        numerical.note,
        numerical.manuscript_term,
    )


def _same_array(current, snapshot):
    if current is None:
        return jnp.asarray(False)
    current = jnp.asarray(current)
    if current.shape != snapshot.shape or current.dtype != snapshot.dtype:
        return jnp.asarray(False)
    return jnp.array_equal(current, snapshot)


class CertifiedHighOrderBasis(SpectralBasis):
    """Basis whose response refuses changed coefficients or certificate inputs.

    The snapshots are JAX leaves so the exact equality check works under JIT.
    A routine ``eqx.tree_at`` edit changes the live field, while its certified
    snapshot remains the one validated at attachment.  This guard detects
    accidental stale-certificate reuse; it is not a tamper-resistant seal
    against deliberate replacement of both live fields and snapshots.
    """

    _certified_matrix: jax.Array
    _certified_error: jax.Array
    _certified_inputs: tuple[jax.Array, ...]
    _certified_static: tuple = eqx.field(static=True)
    _STALE_MESSAGE = "certified high-order basis changed after numerical certification"

    def __init__(self, source: SpectralBasis):
        if (
            source.kernel_terms.numerical.kind != "bound"
            or source.kernel_terms.numerical.value is None
        ):
            raise ValueError("certified basis requires a valued numerical bound")
        matrix = np.asarray(source.response_matrix())
        if dict(source.provenance.numerics).get("certified_matrix") != _matrix_identity(
            matrix
        ):
            raise ValueError("certified basis matrix does not match provenance")
        self.index = source.index
        self.truncation = source.truncation
        self.channels = source.channels
        self.reference = source.reference
        self.support = source.support
        self.I_basis = source.I_basis
        self.V_basis = source.V_basis
        self.P_basis = source.P_basis
        self.kernel_terms = source.kernel_terms
        self.provenance = source.provenance
        self.certified_orders = source.certified_orders
        self.phase = source.phase
        self._certified_matrix = jnp.array(matrix, copy=True)
        self._certified_error = jnp.array(
            source.kernel_terms.numerical.value, copy=True
        )
        self._certified_inputs = tuple(
            jnp.array(value, copy=True) for value in _dynamic_inputs(source)
        )
        self._certified_static = _static_identity(source)

    def _certificate_mismatch(self, matrix):
        """Traced scalar predicate shared by matrix and numerical-budget checks."""
        changed = not _static_identity(self) == self._certified_static
        changed = changed | ~_same_array(matrix, self._certified_matrix)
        changed = changed | ~_same_array(
            self.kernel_terms.numerical.value, self._certified_error
        )
        inputs = _dynamic_inputs(self)
        if len(inputs) != len(self._certified_inputs):
            changed = True
        else:
            for current, snapshot in zip(inputs, self._certified_inputs, strict=True):
                changed = changed | ~_same_array(current, snapshot)
        return changed

    def response_matrix(self) -> jax.Array:
        """The exact cached C, after a JAX-compatible certificate identity check."""
        matrix = super().response_matrix()
        return eqx.error_if(
            matrix, self._certificate_mismatch(matrix), self._STALE_MESSAGE
        )

    def validated_numerical_term(self) -> ErrorTerm:
        """Bound whose value carries its own guard for JIT budget-only outputs.

        ``predict`` can use this instead of the raw stored term so a compiler
        cannot drop the certificate check when callers request only the
        numerical budget and discard the Stokes matrix contraction.
        """
        numerical = self.kernel_terms.numerical
        if numerical.value is None:
            raise ValueError(self._STALE_MESSAGE)
        matrix = super().response_matrix()
        checked = eqx.error_if(
            numerical.value, self._certificate_mismatch(matrix), self._STALE_MESSAGE
        )
        return ErrorTerm(
            checked, numerical.kind, numerical.note, numerical.manuscript_term
        )


def _finite_binary(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite binary float")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(
            f"{name} must be a finite {'positive ' if positive else ''}binary float"
        )
    return result


def _cap(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _column_atol(value, n_real):
    if value is None:
        return None
    try:
        raw = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "column_atol must be a finite nonnegative scalar or column vector"
        ) from exc
    if raw.ndim == 0:
        raw = np.full(n_real, float(raw))
    if raw.shape != (n_real,) or np.any(~np.isfinite(raw)) or np.any(raw < 0):
        raise ValueError(
            "column_atol must be a finite nonnegative scalar or column vector"
        )
    return tuple(float(x) for x in raw)


def _require_current_source(basis):
    if not isinstance(basis, SpectralBasis):
        raise ValueError("basis must be a SpectralBasis")
    if basis.truncation.N > 1:
        raise ValueError("coarse high-order certificate requires Truncation.N <= 1")
    if basis.channels.family != "bump" or basis.channels.normalisation != "unit_peak":
        raise ValueError(
            "coarse high-order certificate requires unit-peak bump channels"
        )
    if not isinstance(basis.phase, TaylorPhase):
        raise ValueError("coarse high-order certificate requires TaylorPhase")
    check_phase(basis.phase, basis.truncation)
    expected = MomentIndex.build(basis.truncation, components=("I", "Q", "V"))
    if (
        basis.index.h0 != expected.h0
        or basis.index.h2 != expected.h2
        or basis.index.n_real != expected.n_real
        or basis.index.pairs != expected.pairs
    ):
        raise ValueError("basis index does not match high-order N<=1 layout")
    provenance = basis.provenance
    kernel = dict(provenance.kernel)
    quadrature = dict(provenance.quadrature)
    if (
        kernel.get("name") != "high_order_harmonic"
        or kernel.get("route") != "host_scipy_real_order"
        or kernel.get("components") != ("I", "Q", "V")
        or quadrature.get("route")
        not in ("ridge_positive_mu_parity", "tensor_positive_mu_parity")
        or quadrature.get("convergence") is not False
        or provenance.channels != tuple(basis.channels.describe())
        or provenance.truncation != truncation_record(basis.index)
        or provenance.reference != tuple(basis.reference.describe())
        or provenance.support != tuple(basis.support.describe())
        or provenance.phase_route != tuple(basis.phase.describe())
        or provenance.assumptions != tuple(basis.phase.forced_assumptions())
        or provenance.units != UNITS
        or provenance.certified_orders != basis.certified_orders
    ):
        raise ValueError("high-order provenance does not match the current basis")
    if basis.kernel_terms.numerical.kind != "unbounded":
        raise ValueError("basis numerical term must be the original unbounded value")


def _matrix_identity(matrix):
    if matrix.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("stored response matrix must use float32 or float64")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("stored response matrix must be finite")
    digest = hashlib.sha256(np.ascontiguousarray(matrix).view(np.uint8)).hexdigest()
    return tuple(matrix.shape), str(matrix.dtype), digest


def _outward_cast(upper, dtype):
    """Publish an MP nonnegative upper bound in the stored matrix dtype."""
    if not upper >= 0 or not upper < upper.ctx.inf:
        raise ArithmeticError("coefficient error is not finite and nonnegative")
    if upper == 0:
        return dtype.type(0)
    nearest = dtype.type(float(upper.b))
    if not np.isfinite(nearest):
        raise ArithmeticError("coefficient error cannot be represented in matrix dtype")
    outward = np.nextafter(nearest, dtype.type(np.inf))
    if not np.isfinite(outward):
        raise ArithmeticError("coefficient error cannot be rounded outward")
    return outward


def _coefficient_error(ctx, binary_value, lower, upper):
    value = ctx.mpf(float(binary_value))
    return max(abs(value - lower).b, abs(upper - value).b)


def certify_high_order_basis_coarse(
    basis: SpectralBasis,
    *,
    i00_atol,
    column_atol=None,
    max_cells: int = 512,
    max_evaluated_cells: int = 1024,
    max_mode_blocks: int = 900,
    max_blocks_per_cell: int = 4,
    decimal_digits: int = 70,
) -> tuple[SpectralBasis, CoarseHighOrderBasisReport]:
    """Attach a coarse fixed-reference numerical envelope after offline work.

    ``column_atol`` is optional and may be one nonnegative number or one per
    stored column. Every row of each column must meet it. Any unsupported
    input, missing I00 certificate, resource cap or unrepresentable outward
    rounded value raises; the supplied immutable basis is untouched. At N=1,
    whole-angle derivative magnitude bounds enclose the extra columns. No
    gamma/B displacement remainder is inferred from this certificate.
    """
    started = perf_counter()
    _require_current_source(basis)
    i00_atol = _finite_binary(i00_atol, "i00_atol", positive=True)
    max_cells = _cap(max_cells, "max_cells", 2)
    max_evaluated_cells = _cap(max_evaluated_cells, "max_evaluated_cells", 2)
    max_mode_blocks = _cap(max_mode_blocks, "max_mode_blocks", 0)
    max_blocks_per_cell = _cap(max_blocks_per_cell, "max_blocks_per_cell", 1)
    decimal_digits = _cap(decimal_digits, "decimal_digits", 50)
    tolerances = _column_atol(column_atol, basis.index.n_real)
    matrix = np.asarray(basis.response_matrix())
    shape, dtype_name, digest = _matrix_identity(matrix)
    if shape != (4 * basis.channels.n_ch, basis.index.n_real):
        raise ValueError("stored response matrix has an invalid shape")
    dtype = matrix.dtype
    gamma = _finite_binary(
        np.asarray(basis.reference.gamma0).item(), "gamma", positive=True
    )
    B = _finite_binary(np.asarray(basis.reference.B0).item(), "B", positive=True)
    if gamma <= 1:
        raise ValueError("gamma must exceed one")
    s_depth = _finite_binary(
        np.asarray(basis.reference.scales[2]).item(), "s_depth", positive=True
    )
    s_gamma = _finite_binary(
        np.asarray(basis.reference.scales[0]).item(), "s_gamma", positive=True
    )
    s_B = _finite_binary(
        np.asarray(basis.reference.scales[1]).item(), "s_B", positive=True
    )
    depth_ref = _finite_binary(
        np.asarray(basis.reference.depth_ref).item(), "depth_ref"
    )
    first_derivatives = any(
        r + s > 0 for _, _, r, s, _ in basis.index.h0 + basis.index.h2
    )

    from mpmath.ctx_iv import MPIntervalContext  # type: ignore[import-untyped]

    ctx = MPIntervalContext()
    ctx.dps = decimal_digits
    envelope = np.zeros(shape, dtype=dtype)
    required_errors = []
    mass = basis.index.position(0, 0, 0, 0, 0, 0)
    certificates: list[AngularI00Certificate] = []
    for channel in range(basis.channels.n_ch):
        row = 4 * channel
        certificate = certify_angular_i00(
            _single_channel(basis.channels, channel),
            gamma,
            B,
            float(matrix[row, mass]),
            atol=i00_atol,
            max_cells=max_cells,
            max_evaluated_cells=max_evaluated_cells,
            max_mode_blocks=max_mode_blocks,
            max_blocks_per_cell=max_blocks_per_cell,
            decimal_digits=decimal_digits,
            first_derivatives=first_derivatives,
            phase=basis.phase if first_derivatives else None,
            depth_ref=depth_ref,
            s_depth=s_depth,
        )
        certificates.append(certificate)
        # The angular routine owns a separate MP context.  Copy its exact
        # dyadic endpoints into this context and check the conversion before
        # combining them with new interval constants.
        i00 = ctx.mpf(certificate.interval)
        if i00.a > certificate.interval.a or i00.b < certificate.interval.b:
            raise ArithmeticError("I00 interval context conversion rounded inward")
        if i00.a < 0 or not i00.b < ctx.inf:
            raise ArithmeticError("I00 certificate must be finite and nonnegative")
        support_lo = float(np.asarray(basis.channels.support)[channel, 0])
        if not (math.isfinite(support_lo) and support_lo > 0):
            raise ValueError("channel support lower frequency must be positive")
        tau_max = (2 * (ctx.mpf(C_SI_M) / ctx.mpf(support_lo)) ** 2).b
        weights = [ctx.mpf(1)]
        for degree in range(1, basis.truncation.max_b() + 1):
            weights.append((weights[-1] * tau_max * ctx.mpf(s_depth) / degree).b)
        derivative_upper = certificate.derivative_upper
        if first_derivatives and derivative_upper is None:
            raise ArithmeticError("N=1 certificate omitted derivative bounds")
        for entry in basis.index.entries():
            norm = (2 * entry.l + 1) * (2 * entry.k + 1)
            derivative = entry.r + entry.s
            if derivative == 0:
                row_upper = i00.b * (1 if entry.h == 0 else weights[entry.b])
            else:
                direction = 0 if entry.r == 1 else 1
                phase_row = 0 if entry.h == 0 else entry.b + 1
                scale = s_gamma if direction == 0 else s_B
                row_upper = (
                    derivative_upper[direction][phase_row].b * ctx.mpf(scale)
                ).b
            if entry.h == 0:
                if (entry.l + entry.k) % 2 == 0:
                    stokes = 0
                else:
                    stokes = 3
                allowed = (stokes,)
                coefficient_upper = (row_upper * norm).b
            else:
                allowed = (1, 2)
                coefficient_upper = (row_upper * norm).b
            for stokes in range(4):
                stored = matrix[row + stokes, entry.slot]
                if stokes not in allowed:
                    # These are exact model zeros and must remain exact zeros
                    # in the fast matrix; a stale or altered basis fails closed.
                    if stored != 0:
                        raise ValueError("stored response violates a structural zero")
                    continue
                if (
                    derivative == 0
                    and entry.h == 0
                    and entry.l == entry.k == 0
                    and stokes == 0
                ):
                    lower, upper = i00.a, i00.b
                else:
                    lower, upper = -coefficient_upper, coefficient_upper
                error = _coefficient_error(ctx, stored, lower, upper)
                envelope[row + stokes, entry.slot] = _outward_cast(error, dtype)
                required_errors.append((row + stokes, entry.slot, error.b))

    note = (
        "Conditional coarse N<=1 numerical coefficient bound for fixed binary "
        "reference/channel inputs and ideal unit-peak bump; relies on experimental "
        "mpmath interval arithmetic. Excludes physical discrepancy, population "
        "tail and gamma/B displacement remainder. At N=1 the derivative "
        "magnitudes use coarse whole-angle Bessel bounds."
    )
    numerical = ErrorTerm(jnp.asarray(envelope, dtype=dtype), "bound", note, "E_num")
    published = np.asarray(numerical.value)
    if (
        published.shape != shape
        or published.dtype != dtype
        or np.any(~np.isfinite(published))
    ):
        raise ArithmeticError(
            "stored numerical envelope changed shape, dtype or finiteness"
        )
    if np.any(published < envelope):
        raise ArithmeticError("stored numerical envelope rounded inward")
    for i, j, required in required_errors:
        if ctx.mpf(float(published[i, j])).a < required:
            raise ArithmeticError("stored numerical envelope does not enclose MP error")
    if tolerances is not None:
        over = np.argwhere(published > np.asarray(tolerances)[None, :])
        if over.size:
            i, j = (int(v) for v in over[0])
            raise ArithmeticError(
                f"column_atol unmet at matrix entry ({i}, {j}): "
                f"bound={published[i, j]:.6g}, atol={tolerances[j]:.6g}"
            )
    config = (
        ("i00_atol", i00_atol),
        ("first_derivatives", first_derivatives),
        ("column_atol", tolerances),
        ("max_cells", max_cells),
        ("max_evaluated_cells", max_evaluated_cells),
        ("max_mode_blocks", max_mode_blocks),
        ("max_blocks_per_cell", max_blocks_per_cell),
        ("decimal_digits", decimal_digits),
    )
    provenance = dataclasses.replace(
        basis.provenance,
        numerics=basis.provenance.numerics
        + (
            (
                "coarse_i00_certificate",
                ("conditional_experimental_mpmath", config),
            ),
            ("certified_matrix", (shape, dtype_name, digest)),
        ),
        notes=basis.provenance.notes + (note,),
    )
    terms = KernelTerms(
        physical_kernel=basis.kernel_terms.physical_kernel,
        harmonic_truncation=basis.kernel_terms.harmonic_truncation,
        numerical=numerical,
        screen_exponent=basis.kernel_terms.screen_exponent,
    )
    attached = SpectralBasis(
        index=basis.index,
        truncation=basis.truncation,
        channels=basis.channels,
        reference=basis.reference,
        support=basis.support,
        I_basis=basis.I_basis,
        V_basis=basis.V_basis,
        P_basis=basis.P_basis,
        kernel_terms=terms,
        provenance=provenance,
        certified_orders=basis.certified_orders,
        phase=basis.phase,
    )
    certified = CertifiedHighOrderBasis(attached)
    if not np.array_equal(np.asarray(certified.response_matrix()), matrix):
        raise ArithmeticError("certificate attachment changed the fast response matrix")
    maximum_column_error = tuple(float(v) for v in np.max(published, axis=0))
    report = CoarseHighOrderBasisReport(
        shape,
        dtype_name,
        digest,
        config,
        tuple(certificates),
        maximum_column_error,
        perf_counter() - started,
    )
    return certified, report


__all__ = ["CoarseHighOrderBasisReport", "certify_high_order_basis_coarse"]
