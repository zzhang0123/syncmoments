"""Harmonic (Dirac-line) channel kernel (``syncmoments.model.harmonic``).

LABEL: ``eq: smooth channel kernel`` (``H_{P,j} = sum_m Q_m R_j(nu_m)
exp(2 i (c/nu_m)^2 varphi)``, ``nu_m = m nu_B / D``; ``I, V`` without phase),
``eq: channel derivative coefficients`` (``build_basis`` uses the Taylor
tensors of :meth:`HarmonicKernel.angular_taylor`), ``extra eq: channel kernel`` and
``extra eq: faraday channel kernel``. The natural-basis line powers
``(I_m, Q_m, V_m)`` are the recurrence used by ``syncmoments.stokes``
(``J_m/sin theta = b_perp (J_{m-1} + J_{m+1})/(2D)``), in erg/s/sr per
electron, at a traced float harmonic index with a static Bessel resolution
``n_nodes`` (default ``2^ceil(log2(max(128, 4 m_max + 64)))``, the automatic
count of ``syncmoments.bessel``); ``J_{m-1}``, ``J_{m+1}`` and ``J_m'`` share
one contour quadrature per point (``bessel_jn_neighbours``).

Memory: one ``lax.map`` step of ``angular_projection``, ``angular_taylor``
and the tail probe holds at most ``chunk_budget`` Bessel integrand values
(``chunk_plan``: up to ``m_chunk`` harmonics by ``vmap``, angular points in
blocks). ``channel_modes`` plans for one point; vmapped over ``S`` samples
it holds ``S`` times that, so ``direct_channel_average`` caps ``S`` per step
by ``samples_per_step``. Blocking changes only the summation order.

Derivatives (``derivatives="analytic"``, default): ``angular_taylor`` gives
the ``(z_gamma, z_B)`` Taylor tensors for ``build_basis`` with no tangent
through the Bessel quadrature (order recurrence on the same contour;
Leibniz-assembled Legendre contraction, ``_harmonic_taylor``);
``angular_projection`` differentiates the Bessel triple by the recurrence.
``"autodiff"`` nests ``jacfwd`` through the quadrature and the contraction
(the v0.2.0 structure on the shared-contour rule ``bessel_jn_neighbours``,
not bit-identical to v0.2.0; its Bessel derivatives lose accuracy at
``|x| << m``). Both are the same AD of the same program and agree to
roundoff (tests: ``1e-12`` per Stokes block).

Angular projection, primary route ``quadrature="product"`` (a JAX port of the
manuscript's ``validation/full_response_product.py``, see
``_harmonic_cells.product_cells``): for each harmonic, channel and sign a cell
in ``t = mu eta`` bounded by the channel support, ``t = sign v^4`` on
Gauss-Legendre nodes, ``log mu`` on Gauss-Legendre nodes, and the ``mu < 0``
half through the parity factors ``(1 +/- (-1)^(l+k))``. No Python branch
depends on a traced value. The response vanishes to all orders at the support
edges, so the moving cell boundaries contribute no boundary terms. AD
differentiates the rule itself except in cells with ``0 < lo < 0.1 hi``,
whose nodes follow the bounds affinely in ``t``, and at an exact
channel-edge/line coincidence at ``t = 0``, where the rule is not
differentiable (``product_cells``); there AD is not the derivative of the
computed values, and either way a derivative carries a quadrature error of
its own (estimated by ``numerical``, not bounded).
Secondary route ``quadrature="tensor"``: ``n_mu x n_eta``
Gauss-Legendre in ``(mu, eta)``; it converges slowly near the ``mu eta = 0``
resonance and serves the finite route check of ``basis.py``.

Shapes: scalar ``(gamma, B, mu, eta)``; ``Modes`` ``(n_ch,)``/``(n_weights, n_ch)``;
``ProjectedModes`` over ``MomentIndex.pairs``. Units: Gauss, Hz, m^2 for ``tau``.
Regime: ``gamma <~ 50``, ``m_max <~ 1e3``; Galactic harmonic numbers
(``m ~ 1e11``) are outside the design and refused by ``build_basis``.
``mode_intervals`` selects inclusive ranges before the Bessel evaluation;
``build_basis`` also prunes orders excluded by the declared gamma/B and channel
supports. This does not relax the contour resolution or high-order guard.

Not certified: quadrature error of either angular route (the refined-rule
``numerical`` estimate and the route check live in ``basis.py``), Bessel roundoff beyond the
finite resolution-class checks in the tests, and the physics of the vacuum
helical-orbit, pure-rotation reference (``physical_error`` is ``unbounded``
unless ``E_phys`` is declared).
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._harmonic_cells import (
    CHUNK_BUDGET,
    auto_nodes,
    checked_nodes,
    chunk_plan,
    harmonic_lines,
    product_cells,
    static_int,
    sum_harmonics,
)
from ._harmonic_tail import probe_estimate
from ._harmonic_ranges import (
    intersect_intervals,
    normalise_intervals,
    subtract_intervals,
    support_intervals,
)
from ._harmonic_taylor import ROUTES, lower_closure, multi_indices
from ._kernel_helpers import (
    Modes,
    ProjectedModes,
    check_scalar_point,
    gather_pairs,
    gyrofrequency_hz,
    legendre_norm,
    phase_coordinate,
    phase_weights,
    resolve_index,
)
from ._support_check import hypothesis_note, outside_support
from .errors import ErrorTerm
from .kernels import required_m_max

QUADRATURES = ("product", "tensor")
DERIVATIVES = ("analytic", "autodiff")


class HarmonicKernel(eqx.Module):
    """Dirac-line harmonic channel kernel; see the module docstring for the contract.

    Static fields: ``m_max`` (largest harmonic considered), ``mode_intervals``
    (inclusive selected ranges; ``None`` means ``1..m_max``), ``n_nodes`` (Bessel
    resolution, ``None`` for the automatic count; an explicit count must be
    ``>= 4 m_max + 2``), ``n_outer``/``n_inner`` (product-cell Gauss-Legendre
    counts), ``n_mu``/``n_eta`` (tensor route), ``m_chunk`` (harmonics
    batched by ``vmap`` inside one ``lax.map`` step, an upper cap),
    ``chunk_budget`` (Bessel integrand values per ``lax.map`` step, default
    ``CHUNK_BUDGET = 2^20``; module docstring, :meth:`samples_per_step`),
    ``tail_probe`` (harmonics beyond ``m_max`` summed for the truncation
    estimate, or ``None``), ``quadrature`` (default route), ``derivatives``
    (``"analytic"`` or ``"autodiff"``; module docstring). ``E_phys`` is an
    optional declared physical-error term (leaf). Units: per-electron channel
    Stokes; ``B`` in Gauss, Hz. Assumes vacuum helical-orbit radiation with a
    pure-rotation phase; not certified: that model (``physical_error``
    unbounded unless ``E_phys`` is declared) and the quadrature beyond the
    ``numerical`` estimate.
    """

    m_max: int = eqx.field(static=True)
    mode_intervals: tuple[tuple[int, int], ...] | None = eqx.field(static=True)
    channel_mode_intervals: tuple[tuple[tuple[int, int], ...], ...] | None = eqx.field(
        static=True
    )
    selected_channel_support: tuple[tuple[float, float], ...] | None = eqx.field(
        static=True
    )
    selected_parameter_support: tuple[tuple[float, float], ...] | None = eqx.field(
        static=True
    )
    n_nodes: int | None = eqx.field(static=True)
    n_outer: int = eqx.field(static=True)
    n_inner: int = eqx.field(static=True)
    n_mu: int = eqx.field(static=True)
    n_eta: int = eqx.field(static=True)
    m_chunk: int = eqx.field(static=True)
    chunk_budget: int = eqx.field(static=True)
    tail_probe: int | None = eqx.field(static=True)
    E_phys: ErrorTerm | None
    quadrature: str = eqx.field(static=True)
    derivatives: str = eqx.field(static=True)

    name: ClassVar[str] = "harmonic"
    LABEL: ClassVar[tuple[str, ...]] = (
        "eq: smooth channel kernel",
        "eq: channel derivative coefficients",
        "extra eq: channel kernel",
        "extra eq: faraday channel kernel",
    )
    components: ClassVar[tuple[str, ...]] = ("I", "Q", "V")
    required_closures: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        m_max,
        *,
        mode_intervals=None,
        n_nodes=None,
        n_outer=64,
        n_inner=64,
        n_mu=48,
        n_eta=48,
        m_chunk=64,
        chunk_budget=CHUNK_BUDGET,
        tail_probe=None,
        E_phys=None,
        quadrature="product",
        derivatives="analytic",
    ):
        m_max = static_int(m_max, "m_max", 1)
        n_nodes = checked_nodes(m_max, n_nodes)
        if quadrature not in QUADRATURES:
            raise ValueError(f"quadrature must be one of {QUADRATURES}")
        if derivatives not in DERIVATIVES:
            raise ValueError(f"derivatives must be one of {DERIVATIVES}")
        if E_phys is not None and not isinstance(E_phys, ErrorTerm):
            raise ValueError("E_phys must be an ErrorTerm or None")
        self.m_max = m_max
        self.mode_intervals = (
            None
            if mode_intervals is None
            else normalise_intervals(mode_intervals, m_max)
        )
        self.channel_mode_intervals = None
        self.selected_channel_support = None
        self.selected_parameter_support = None
        self.n_nodes = n_nodes
        self.n_outer = static_int(n_outer, "n_outer", 2)
        self.n_inner = static_int(n_inner, "n_inner", 2)
        self.n_mu = static_int(n_mu, "n_mu", 2)
        self.n_eta = static_int(n_eta, "n_eta", 2)
        self.m_chunk = static_int(m_chunk, "m_chunk", 1)
        self.chunk_budget = static_int(chunk_budget, "chunk_budget", 1)
        self.tail_probe = (
            None if tail_probe is None else static_int(tail_probe, "tail_probe", 1)
        )
        self.E_phys = E_phys
        self.quadrature = quadrature
        self.derivatives = derivatives

    # -- harmonic batching -----------------------------------------------------

    def resolution(self) -> int:
        """The static Bessel node count in use."""
        return auto_nodes(self.m_max) if self.n_nodes is None else self.n_nodes

    def harmonics(self) -> jax.Array:
        """Selected harmonic orders as floats; may be empty or disjoint."""
        intervals = self.selected_intervals()
        if not intervals:
            return jnp.empty((0,), dtype=float)
        return jnp.concatenate(
            [jnp.arange(lo, hi + 1, dtype=float) for lo, hi in intervals]
        )

    def selected_intervals(self) -> tuple[tuple[int, int], ...]:
        """Inclusive order ranges actually sent to the Bessel evaluator."""
        return (
            ((1, self.m_max),) if self.mode_intervals is None else self.mode_intervals
        )

    def selected_mode_count(self) -> int:
        """Number of selected integers, without constructing a mode array."""
        return sum(hi - lo + 1 for lo, hi in self.selected_intervals())

    def _copy(self, **changes) -> "HarmonicKernel":
        """Rebuild while retaining the channel binding of a pruned kernel."""
        values = dict(
            m_max=self.m_max,
            mode_intervals=self.mode_intervals,
            n_nodes=self.n_nodes,
            n_outer=self.n_outer,
            n_inner=self.n_inner,
            n_mu=self.n_mu,
            n_eta=self.n_eta,
            m_chunk=self.m_chunk,
            chunk_budget=self.chunk_budget,
            tail_probe=self.tail_probe,
            E_phys=self.E_phys,
            quadrature=self.quadrature,
            derivatives=self.derivatives,
        )
        values.update(changes)
        clone = HarmonicKernel(**values)
        object.__setattr__(clone, "channel_mode_intervals", self.channel_mode_intervals)
        object.__setattr__(
            clone, "selected_channel_support", self.selected_channel_support
        )
        object.__setattr__(
            clone, "selected_parameter_support", self.selected_parameter_support
        )
        return clone

    def for_support(self, support, channels, *, reference=None) -> "HarmonicKernel":
        """Keep only modes that can meet a channel on the declared support.

        Pruning is exact for the mathematical line model provided all emitting
        electrons lie inside the declared gamma/B support.  This method does
        not certify that declaration or the floating-point line evaluation.
        The returned kernel is tied to these channels' ordered support edges.
        """
        self._check_channel_support_eager(channels)
        gamma_bounds = tuple(map(float, support.gamma))
        B_bounds = tuple(map(float, support.B))
        if reference is not None:
            gamma_ref = float(np.asarray(reference.gamma0))
            B_ref = float(np.asarray(reference.B0))
            gamma_bounds = (
                min(gamma_bounds[0], gamma_ref),
                max(gamma_bounds[1], gamma_ref),
            )
            B_bounds = (min(B_bounds[0], B_ref), max(B_bounds[1], B_ref))
        parameter_support = (gamma_bounds, B_bounds)
        if self.selected_parameter_support is not None and any(
            new[0] < old[0] or new[1] > old[1]
            for new, old in zip(parameter_support, self.selected_parameter_support)
        ):
            raise ValueError(
                "support-pruned kernel cannot be widened; start from an unpruned "
                "HarmonicKernel for a larger gamma/B support"
            )
        by_channel = support_intervals(
            support, channels, self.m_max, reference, separate=True
        )
        selected_by_channel = tuple(
            intersect_intervals(self.selected_intervals(), ranges)
            for ranges in by_channel
        )
        selected = normalise_intervals(
            [interval for ranges in selected_by_channel for interval in ranges],
            self.m_max,
        )
        pruned = self._copy(mode_intervals=selected)
        object.__setattr__(pruned, "channel_mode_intervals", selected_by_channel)
        object.__setattr__(
            pruned,
            "selected_channel_support",
            tuple(tuple(map(float, row)) for row in np.asarray(channels.support)),
        )
        object.__setattr__(pruned, "selected_parameter_support", parameter_support)
        return pruned

    def _point_outside_selection(self, gamma, B):
        if self.selected_parameter_support is None:
            return jnp.asarray(False)
        (g_lo, g_hi), (B_lo, B_hi) = self.selected_parameter_support
        return (gamma < g_lo) | (gamma > g_hi) | (B < B_lo) | (B > B_hi)

    def _checked_point(self, gamma, B):
        if self.selected_parameter_support is None:
            return gamma, B
        gamma = eqx.error_if(
            gamma,
            self._point_outside_selection(gamma, B),
            "support-pruned kernel is tied to its original gamma/B support",
        )
        return gamma, B

    def _check_channel_support_eager(self, channels) -> None:
        """Reject reuse of a channel-specific selection with another band layout."""
        if self.selected_channel_support is None:
            return
        actual = np.asarray(channels.support, dtype=float)
        expected = np.asarray(self.selected_channel_support, dtype=float)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                "support-pruned kernel is tied to its original channel supports; "
                "start from an unpruned HarmonicKernel for different channels"
            )

    def _checked_channels(self, channels):
        """JAX-safe channel-support check for the public forward routes."""
        if self.selected_channel_support is None:
            return channels
        expected = jnp.asarray(self.selected_channel_support)
        if channels.support.shape != expected.shape:
            raise ValueError("support-pruned kernel has a different channel count")
        checked = eqx.error_if(
            channels.support,
            jnp.any(channels.support != expected),
            "support-pruned kernel is tied to its original channel supports",
        )
        return dataclasses.replace(channels, support=checked)

    def _guard_empty_output(self, value, channels, gamma, B):
        """Keep selection checks live when no harmonic is evaluated."""
        if self.selected_channel_support is None:
            return value
        return eqx.error_if(
            value,
            jnp.any(channels.support != jnp.asarray(self.selected_channel_support))
            | self._point_outside_selection(gamma, B),
            "support-pruned kernel is tied to its original channel or gamma/B support",
        )

    def missing_intervals(
        self, support, channels, *, reference=None
    ) -> tuple[tuple[int, int], ...]:
        """Potentially active orders ``<= m_max`` omitted by manual selection."""
        self._check_channel_support_eager(channels)
        missing = list(
            subtract_intervals(
                support_intervals(support, channels, self.m_max, reference),
                self.selected_intervals(),
            )
        )
        if self.channel_mode_intervals is not None:
            required_by_channel = support_intervals(
                support, channels, self.m_max, reference, separate=True
            )
            if len(self.channel_mode_intervals) != len(required_by_channel):
                raise ValueError(
                    "channel_mode_intervals must have one entry per channel"
                )
            missing.extend(
                interval
                for required, selected in zip(
                    required_by_channel, self.channel_mode_intervals
                )
                for interval in subtract_intervals(required, selected)
            )
        return normalise_intervals(missing, self.m_max)

    def _rule(self) -> str:
        return "recurrence" if self.derivatives == "analytic" else "autodiff"

    def _plan(self, points: int, tangents: int = 1) -> tuple[int, int]:
        """``(harmonics, block)`` per ``lax.map`` step for ``points`` angular points;
        ``tangents`` forward tangents through the Bessel contour count
        ``n_nodes`` values each per point."""
        per_point = self.resolution() * max(1, int(tangents))
        return chunk_plan(points, per_point, self.m_chunk, self.chunk_budget)

    def for_tangents(self, tangents: int) -> "HarmonicKernel":
        """Copy with ``chunk_budget // tangents``: for outer nested ``jacfwd``
        through the Bessel contour (``"autodiff"`` basis path), where every
        tangent carries ``n_nodes`` values per point. Summation order only."""
        budget = max(1, self.chunk_budget // max(1, int(tangents)))
        return self._copy(chunk_budget=budget)

    def samples_per_step(self, channels, requested: int) -> int:
        """``max(1, min(requested, chunk_budget // (h n_nodes)))``: samples per step
        of a ``vmap`` of ``channel_modes`` (``h`` harmonics per step, one point)
        within ``chunk_budget``; tangents of an outer ``jacfwd`` are not counted."""
        harmonics = min(self._plan(1)[0], max(1, self.selected_mode_count()))
        per_sample = harmonics * self.resolution()
        return max(1, min(int(requested), self.chunk_budget // per_sample))

    # -- protocol ---------------------------------------------------------------

    def line_frequencies(self, gamma, B, mu, eta) -> jax.Array:
        """``nu_m = m nu_B / (1 - beta mu eta)`` [Hz] for selected modes."""
        gamma, B, mu, eta = check_scalar_point(gamma, B, mu, eta)
        gamma, B = self._checked_point(gamma, B)
        beta = jnp.sqrt(1.0 - 1.0 / gamma**2)
        values = self.harmonics() * gyrofrequency_hz(gamma, B) / (1.0 - beta * mu * eta)
        if (
            not self.selected_intervals()
            and self.selected_parameter_support is not None
        ):
            values = eqx.error_if(
                values,
                self._point_outside_selection(gamma, B),
                "support-pruned kernel is tied to its original gamma/B support",
            )
        return values

    def channel_modes(
        self, channels, gamma, B, mu, eta, *, phase=None, depth_ref=0.0, s_depth=1.0
    ):
        gamma, B, mu, eta = check_scalar_point(gamma, B, mu, eta)
        gamma, B = self._checked_point(gamma, B)
        channels = self._checked_channels(channels)
        n_nodes = self.resolution()
        n_ch = channels.n_ch
        active = (B > 0.0).astype(float)
        if not self.selected_intervals():
            n_weights = 1 if phase is None else phase.n_weights
            zeros = jnp.zeros(n_ch, dtype=jnp.result_type(gamma, B, 1.0))
            return Modes(
                I=self._guard_empty_output(zeros, channels, gamma, B),
                V=self._guard_empty_output(zeros, channels, gamma, B),
                P=self._guard_empty_output(
                    jnp.zeros((n_weights, n_ch), dtype=complex), channels, gamma, B
                ),
            )

        def one(m):
            I, Q, V, nu = harmonic_lines(
                m, gamma, B, mu, eta, n_nodes=n_nodes, bessel=self._rule()
            )
            nu_safe = jnp.where(nu > 0.0, nu, 1.0)
            response = channels(nu_safe) * active  # (n_ch,)
            w = phase_weights(phase, phase_coordinate(nu_safe), depth_ref, s_depth)
            return response * I, response * V, (response * Q)[None, :] * w[:, None]

        I, V, P = sum_harmonics(one, self.harmonics(), self._plan(1)[0])
        return Modes(I=I.reshape(n_ch), V=V.reshape(n_ch), P=P)

    def angular_projection(
        self,
        channels,
        gamma,
        B,
        *,
        phase=None,
        depth_ref=0.0,
        s_depth=1.0,
        truncation,
        quadrature=None,
    ):
        return self.angular_taylor(
            channels,
            gamma,
            B,
            scales=(1.0, 1.0),
            phase=phase,
            depth_ref=depth_ref,
            s_depth=s_depth,
            truncation=truncation,
            quadrature=quadrature,
            order=0,
        )[(0, 0)]

    def angular_taylor(
        self,
        channels,
        gamma0,
        B0,
        *,
        scales,
        phase=None,
        depth_ref=0.0,
        s_depth=1.0,
        truncation,
        quadrature=None,
        order,
        orders=None,
    ) -> dict:
        """``{(r, s): ProjectedModes}``: ``d^r_{z_gamma} d^s_{z_B}`` at ``z = 0``.

        The projection is taken at ``(gamma0 + s_gamma z_gamma, B0 + s_B
        z_B)`` with ``scales = (s_gamma, s_B)``, for ``r + s <= order``
        (static), without ``1/(r! s!)``; ``order = 0`` is
        :meth:`angular_projection`. ``orders`` (static set of ``(r, s)``,
        ``r + s <= order``) restricts the output to its lower closure and
        the nested passes to the directions it needs (per-variable caps).
        Differentiable in every traced leaf (outer ``jacfwd``/``grad``).
        Same units and caveats as the projection; see the module docstring
        for the derivative paths.
        """
        quadrature = self.quadrature if quadrature is None else quadrature
        if quadrature not in QUADRATURES:
            raise ValueError(f"quadrature must be one of {QUADRATURES}")
        order = static_int(order, "order", 0)
        alphas = multi_indices(order) if orders is None else lower_closure(orders)
        if max(r + s for r, s in alphas) > order:
            raise ValueError(f"orders {sorted(alphas)} exceed order={order}")
        index = resolve_index(truncation, self.components)
        gamma0, B0, _, _ = check_scalar_point(gamma0, B0, 0.0, 0.0)
        gamma0, B0 = self._checked_point(gamma0, B0)
        channels = self._checked_channels(channels)
        s_gamma, s_B = scales

        if not self.selected_intervals():
            n_ch, n_pairs = channels.n_ch, len(index.pairs)
            n_weights = 1 if phase is None else phase.n_weights
            zeros = jnp.zeros((n_ch, n_pairs), dtype=jnp.result_type(gamma0, B0, 1.0))
            polarised = jnp.zeros((n_weights, n_ch, n_pairs), dtype=complex)
            return {
                alpha: ProjectedModes(  # type: ignore[call-arg]
                    I=self._guard_empty_output(zeros, channels, gamma0, B0),
                    V=self._guard_empty_output(zeros, channels, gamma0, B0),
                    P=self._guard_empty_output(polarised, channels, gamma0, B0),
                )
                for alpha in alphas
            }

        def lift(z):
            return gamma0 + s_gamma * z[0], B0 + s_B * z[1]

        degrees = (index.truncation.L_mu, index.truncation.L_eta)
        rule = "taylor" if order and self.derivatives == "analytic" else self._rule()
        grids = ROUTES[quadrature](
            self, channels, lift, phase, depth_ref, s_depth, degrees, alphas, rule
        )
        norm = legendre_norm(index.pairs)
        return {
            alpha: ProjectedModes(
                *(gather_pairs(grid, index.pairs) * norm for grid in grids[alpha])
            )
            for alpha in alphas
        }

    def physical_error(self, channels) -> ErrorTerm:
        if self.E_phys is not None:
            return self.E_phys
        return ErrorTerm.unbounded(
            "vacuum helical-orbit, pure-rotation reference (main.tex sec: stokes "
            "harmonic, sec: screen model); zero only if declared",
            "E_phys",
        )

    def truncation_error(
        self, support, channels, *, samples=None, reference=None, phase=None
    ):
        """Omitted-mode term for unselected orders, per electron, ``(n_ch, 4)``.

        ``bound`` zero when ``m_max >= required_m_max(support, channels)``,
        selected intervals cover all potentially active orders, every channel
        support is finite and positive, and no checked emitting sample leaves
        the relevant support (``_support_check``). Manual low/gap omissions
        are ``unbounded``. An upper-tail ``estimate`` comes from probe harmonics
        (``_harmonic_tail``) over ``samples`` or the angular grid at
        ``reference``; other unresolved cases are ``unbounded``. Columns are
        ``I, Q, U, V`` with the natural ``|Q_m|`` sum in both ``Q`` and ``U``.
        """
        n_ch = channels.n_ch
        required = required_m_max(support, channels)
        supp = np.asarray(channels.support, dtype=float)
        compact = bool(np.all(np.isfinite(supp)) and np.all(supp[:, 0] > 0))
        if reference is not None and any(
            isinstance(getattr(reference, name), jax.core.Tracer)
            for name in ("gamma0", "B0")
        ):
            return ErrorTerm.unbounded(
                "traced reference: selected harmonic coverage cannot be checked eagerly",
                "E_num",
            )
        outside = outside_support(
            samples,
            support,
            harmonic_only=self.mode_intervals is None,
            harmonic_range=self.mode_intervals is not None,
            strict=self.mode_intervals is not None,
        )
        missing = self.missing_intervals(support, channels, reference=reference)
        if missing:
            return ErrorTerm.unbounded(
                f"mode_intervals omit potentially active orders {missing} on the "
                "declared support; no lower/gap omission envelope supplied",
                "E_num",
            )
        reference_outside = False
        if reference is not None:
            g_ref, B_ref = float(np.asarray(reference.gamma0)), float(
                np.asarray(reference.B0)
            )
            reference_outside = (
                g_ref > float(np.asarray(support.gamma[1]))
                or B_ref < float(np.asarray(support.B[0]))
                or B_ref > float(np.asarray(support.B[1]))
            )
        if self.mode_intervals is not None and (
            outside or (samples is not None and outside is None) or reference_outside
        ):
            return ErrorTerm.unbounded(
                "selected low/gap orders are justified only inside the declared "
                "Support; the supplied samples or reference leave it or could "
                "not be checked, and an upper-tail probe cannot cover them",
                "E_num",
            )
        if self.m_max >= required and compact and not outside and not reference_outside:
            return ErrorTerm(
                jnp.zeros((n_ch, 4)),
                "bound",
                f"m_max={self.m_max} >= required_m_max={required}; selected "
                "intervals cover every potentially active line on the declared "
                "support; "
                + hypothesis_note(
                    samples,
                    outside,
                    harmonic_range=self.mode_intervals is not None,
                ),
                "E_num",
            )
        reason = f"m_max={self.m_max} < required_m_max={required}"
        if self.m_max >= required and not compact:
            reason = "a channel support is not finite with a positive lower edge"
        if outside:
            reason = (
                f"samples outside the declared Support ({'; '.join(outside)}), so "
                f"required_m_max={required} does not cover them"
            )
        if reference_outside:
            reason = (
                "reference gamma/B lies outside the declared Support, so "
                "required_m_max does not certify the reference evaluation"
            )
        if self.tail_probe is None or (samples is None and reference is None):
            return ErrorTerm.unbounded(
                f"{reason}; set tail_probe and supply samples or a reference for "
                "an estimate, or raise m_max (or widen the Support)",
                "E_num",
            )
        return probe_estimate(self, channels, samples, reference, f"{reason}: ")

    def describe(self) -> tuple:
        return (
            ("name", self.name),
            ("label", self.LABEL),
            ("m_max", self.m_max),
            ("mode_intervals", self.selected_intervals()),
            ("channel_mode_intervals", self.channel_mode_intervals),
            ("selected_channel_support", self.selected_channel_support),
            ("selected_parameter_support", self.selected_parameter_support),
            ("n_nodes", self.resolution()),
            ("quadrature", self.quadrature),
            ("n_outer", self.n_outer),
            ("n_inner", self.n_inner),
            ("n_mu", self.n_mu),
            ("n_eta", self.n_eta),
            ("m_chunk", self.m_chunk),
            ("chunk_budget", self.chunk_budget),
            ("tail_probe", self.tail_probe),
            ("derivatives", self.derivatives),
            ("E_phys", None if self.E_phys is None else self.E_phys.kind),
        )


__all__ = ["HarmonicKernel", "harmonic_lines", "product_cells", "auto_nodes"]
