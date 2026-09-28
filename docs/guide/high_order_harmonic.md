# High-order finite-$\gamma$ harmonic channels

Choose the entry point by the error statement needed downstream:

| Entry point | Calculation | Numerical error statement |
| --- | --- | --- |
| `high_order_channel_modes` | One fixed particle and viewing direction, with separate $I$, $V$ and line-dependent complex $P$ | Exploratory value; the accelerated harmonic-sum, Bessel and roundoff errors are `unbounded`. |
| `build_high_order_basis` | Reusable fixed-reference $N\leq1$ moment response matrix | Exploratory coefficients; `kernel_terms.numerical` is `unbounded`. |
| `certify_high_order_basis_coarse` | Optional offline enclosure of every stored $N\leq1$ matrix coefficient | Conditional fixed-reference per-column `bound`, propagated through `predict`; an unmet tolerance or resource cap raises. |

The certificate assumes an ideal unit-peak `bump` channel, `TaylorPhase`,
fixed exact-binary reference inputs and mpmath's experimental interval
arithmetic. It can be much wider than the signal, especially for $N=1$.
It does not bound a nonzero $\gamma/B$ displacement, omitted physics,
population tails or the final inference error. Those terms must be carried
separately to the intended scientific quantity.

Install the value route with `python -m pip install 'syncmoments[high_harmonic]'`.
Install `mpmath` as well before requesting the optional certificate.

`high_order_channel_modes` evaluates the finite-$\gamma$ discrete-harmonic
reference at one fixed $(\gamma,B,\mu,\eta)$ point. It preserves natural-basis
$I,V,Q$ and applies each line's channel response and Faraday phase before
summing. It is a host-only value calculation and is separate from the JAX
`HarmonicKernel` moment and derivative routes.

## Selected ranges in the existing JAX harmonic kernel

`HarmonicKernel` accepts inclusive `mode_intervals` for a partial sum, for
example `HarmonicKernel(100, mode_intervals=((1, 9), (17, 70)))`. Disjoint
ranges can exclude orders between separated channels. Its `harmonics()` method
shows the orders actually passed to the Bessel evaluator. Without this option,
it uses `1..m_max` for direct `channel_modes` calls.

`build_basis` automatically intersects these ranges with a conservative union
of the channel supports over the declared $\gamma,B$ support and the reference
point. The selected union appears as `computed_mode_intervals` and the separate
product-route selections as `computed_channel_mode_intervals` in the basis
numerics record. The product angular route evaluates each channel's selected
orders; the tensor and direct point routes evaluate the union once and share
each line across channels. Convergence rebuilds retain these selections.
The interval calculation includes one extra order at each endpoint. Its
zero-contribution argument is for the mathematical line frequency and is
conditional on the declared parameter support; floating-point line evaluation
remains uncertified.

If manual `mode_intervals` omit an order that may contribute, `build_basis`
raises unless `allow_truncated=True`. In that case `harmonic_truncation` is
`unbounded`: a small convergence difference does not account for the missing
lines. `for_support(support, channels)` gives a similarly pruned kernel for
direct point evaluations, which should be used only inside that support.
The pruned kernel checks that later channel calls use the same support edges
and channel ordering; create a new selection from the unpruned kernel for a
different band layout. `dataclasses.replace` on `HarmonicKernel` is rejected
because it cannot safely validate the support-bound selection; construct a
new kernel or use its `for_tangents` method for that budget adjustment.
Direct point and angular calls also reject gamma or B outside the range used
for pruning. Reapplying for_support to a pruned kernel cannot widen that
range; start from an unpruned kernel if a larger range is needed.
Range pruning reduces the number of Bessel calls in the validated low-order
kernel. It does not reduce the contour resolution set by `m_max` or enable
the GHz, microgauss high-order `build_basis` path.

## Fixed-point high-order value route

Install the optional `high_harmonic` extra for SciPy. For example:

```python
from syncmoments.model import Channels, TaylorPhase, high_order_channel_modes

channels = Channels.bump([1.0e8], [2.0e7])
result = high_order_channel_modes(
    channels, gamma=3000.0, B=5.0e-6, mu=0.0, eta=0.0,
    phase=TaylorPhase(0), depth_ref=0.1,
)
print(result.modes.I, result.modes.V, result.modes.P)
print(result.routes, result.active_ranges, result.evaluated_orders)
print(result.harmonic_sum.kind, result.quadrature.kind)
```

A finite fixed-point exercise at 0.1, 1 and 3 GHz used 20% half-width `bump`
channels, $\gamma=3000$, $B=5\,\mu\mathrm G$, and $\mu=\eta=0$.
With `max_quad_order=2048`, all three integral quadrature differences met
`rtol=1e-6` in that run; 2,104 real-valued order samples were evaluated rather than
enumerating roughly $10^{10}$–$10^{12}$ integer harmonics. The default
`max_quad_order=256` failed the quadrature check for the 1 and 3 GHz
channels. The accelerated harmonic-sum, Bessel, and floating-point errors
still report `unbounded`, so these values are exploratory.

The channel support determines an inclusive integer range, padded by two
orders against floating endpoint classification. Response values outside the
open support remain zero. For a range with at most `direct_limit` candidate
integers inside the open support, the function sums integer lines directly,
including up to six extra boundary-padding orders. For a denser range it evaluates the
integral plus half of each endpoint from the [Euler–Maclaurin identity, DLMF
2.10.1](https://dlmf.nist.gov/2.10.E1). It uses SciPy's `jv` at
continuous real order for that integral; it does not switch to the $F/G$
continuous-spectrum physical model. The accelerated route is currently
restricted to the package's smooth `bump` response with an order span below
16 times its starting order. A broad one-piece integral can miss low-order
emission even when the response is smooth. Dense `planck_taper` bands are
refused because a sharp internal transition can be missed by both quadrature
rules. Orders beyond $2^{53}-2$ are refused because float64 cannot resolve
unit harmonic spacing.

The sum strategy is selected from the active line count for each channel.
A fixed point has line spacing $s=\nu_B/D$; the candidate count inside the
open support is the number of positive integers in
$(\nu_{j,\mathrm{lo}}/s,\nu_{j,\mathrm{hi}}/s)$, with endpoint padding used
separately for safe evaluation. Thus channel width and the local line spacing
set the work, while the centre frequency alone does not.
A high-order narrow band can take the direct route, while a lower-order dense
band can take the accelerated route. The method for one Bessel line is a
separate decision involving its order and proximity to the turning point.
The current accelerated route lacks a valid absolute error certificate.

## Reusable exploratory moment basis

The host route can prepare a zeroth-order moment basis once:

```python
from syncmoments.model import (
    Channels, Reference, Support, Truncation, build_high_order_basis,
)

channels = Channels.bump([1.0e8], [2.0e7])
reference = Reference(3000.0, 5.0e-6, depth_ref=0.1,
                      scales=(3000.0, 5.0e-6, 1.0))
support = Support((2000.0, 4000.0), (4.0e-6, 6.0e-6), (-1.0, 1.0),
                  truncated=False)
basis = build_high_order_basis(
    channels, Truncation(0, 0, 0), reference, support=support,
    n_mu=24, n_eta=32, atol=1.0e-30,
)
C = basis.response_matrix()
```

This `atol` is an absolute stopping threshold for the *inner exploratory
order integral*, not a bound on a basis coefficient. The default angular
rule integrates over the full viewing-angle domain using
$\eta=\mu+h_j(\mu)\sinh t$, with a separate ridge scale for each channel
and up to three panels per positive pitch-angle node. The positive-$\mu$
parity reconstructs the negative half. A plain `angular_rule="tensor"`
remains available for low-order comparisons; at high $\gamma$ it can miss
the narrow angular ridge. The method rejects failed inner stopping checks
unless `allow_unconverged=True`, which records their count for exploratory
diagnostics. Neither setting certifies angular quadrature, the dense
harmonic-sum remainder, Bessel evaluation or floating-point arithmetic.

For this one-channel example, an independent run produced
$C_{I,0}=1.4167550407\times10^{-21}$ in about 1.06 s for the first build
after imports. It evaluated 1,152 angular nodes and 128,128 real order
samples, with no array sized by the largest active order
($\sim5.1\times10^{10}$); every inner stopping check passed. An independent
physical-$\eta$ adaptive quadrature at the same 24 $\mu$ nodes gave
$1.4167587410\times10^{-21}$, about 2.6 ppm higher and taking about 7.25 s
for the inner angular integrals. This is finite numerical evidence, not a
global error bound. This zeroth-order basis does not represent a displaced
$\gamma/B$ population without a separate Taylor remainder description.

First gamma/B Taylor columns can also be prepared once. The derivative of
each finite-gamma line includes the moving channel response and the
line-frequency-dependent Faraday phase. The two derivative columns are
multiplied by `reference.scales[0:2]` to match the dimensionless moment
coordinates. For example, a 100 MHz high-$\gamma$ basis without a depth
Taylor row is:

```python
first_order = build_high_order_basis(
    channels, Truncation(0, 0, 1, depth_degree=0), reference,
    support=support, n_mu=24, n_eta=32,
    rtol=1e-5, atol=1e-30,
    atol_gamma=1e-27, atol_B=1e-25,
)
```

`atol_gamma` and `atol_B` are stopping thresholds for *physical-unit*
derivatives, before the reference scales are applied. With the sample's
100 MHz, 20%-half-width channel and $(\gamma,B)=(3000,5\,\mu\mathrm G)$,
this 24-by-32 build took about 1.15 s in a local warm run and evaluated
138,752 continuous-order samples. Its inner checks passed. Adding a depth
Taylor row can require different thresholds or more work because the phase
derivative adds variation; failed checks raise by default. The resulting
`N=1` coefficients report **unbounded** numerical error until the separate
certificate below is requested. Its nonzero-displacement Taylor remainder
remains **unbounded** even after fixed-reference coefficient certification.
`Channels.bump` pads its stored support outward by one floating-point step
so the exact-binary smooth zeros $c-w,c+w$ remain inside it. The derivative
route rejects an altered bump channel whose support clips either zero.

The error-controlled calculation is a separate, explicitly requested route.
The public `high_order_channel_modes` call computes exploratory spectra
without a certified total error bound. Its `rtol` and `atol` only stop
refinement of the integral quadrature; they do not bound the discrete-sum,
Bessel, or floating-point errors. The internal dense interval route instead
requires a positive absolute budget and returns only when its whole-channel
bound meets it. Exploratory values cannot enter a certified downstream
error budget without further validation.

For moment inference, prepare the high-order response matrix once and reuse
it in the likelihood. `build_high_order_basis` constructs an exploratory
`N<=1` angular basis with `numerical.kind="unbounded"`. An optional offline
call can attach a coarse **fixed-reference numerical** column envelope:

```python
from syncmoments.model import certify_high_order_basis_coarse

bounded_basis, report = certify_high_order_basis_coarse(
    basis, i00_atol=1e-18, max_cells=512,
    max_evaluated_cells=1024, max_mode_blocks=900,
)
C = bounded_basis.response_matrix()  # identical to the original cached matrix
E_column = bounded_basis.validated_numerical_term().value
```

This call requires `N<=1`, unit-peak `bump` channels, `TaylorPhase`, and an
unchanged high-order basis provenance. For each channel it certifies the
entire angular-domain intensity coefficient $I_{00}$ at the stored binary
$(\gamma_0,B_0)$, then uses $|V|,|Q+iU|\leq I$ and
$|P_l(\mu)P_k(\eta)|\leq1$ to enclose every retained matrix entry. For a
Legendre pair $(l,k)$, the coarse magnitude is at most
$(2l+1)(2k+1)I_{00}^{\rm upper}$; a depth row $b$ also receives
$[2(c/\nu_{\rm support,lo})^2s_{\rm depth}]^b/b!$. The mass-column $I_{00}$
uses its tighter signed interval. Parity-forbidden and Stokes-layout zeros
are checked exactly. Each envelope entry includes the difference to the
*actual stored binary* response coefficient, then is rounded outward into
its stored float32 or float64 dtype and checked after attachment. The report
records the matrix shape, dtype, SHA-256 digest, certificate configuration,
per-channel work counters, and measured elapsed time. `column_atol` may
optionally require every row of every stored column to meet a scalar or
per-column absolute budget; a resource cap or unmet budget raises without
changing the original basis.

For `N=1`, the same optional call additionally bounds the fixed-reference
$\gamma/B$ derivative columns. It uses the [integer-order Bessel integral,
DLMF 10.9.2](https://dlmf.nist.gov/10.9.E2) to bound neighbouring Bessel
values and their argument derivatives on each angular cell, then includes
the moving bump response and line-dependent Faraday phase derivatives.
Before doing so, it checks with interval arithmetic that the stored channel
support contains both exact-binary bump zeros $c-w$ and $c+w$; otherwise a
moving support edge could introduce an omitted boundary term, so it raises.
The resulting derivative magnitude bound is deliberately coarse. It is
compared with every stored derivative coefficient and propagated as a
per-column numerical error, with no certificate work in `predict`.
The same function accepts the prepared `first_order` basis:

```python
first_order_bounded, first_order_report = certify_high_order_basis_coarse(
    first_order, i00_atol=2e-18,
    max_cells=512, max_evaluated_cells=1024, max_mode_blocks=900,
)
```

The `bound` classification is conditional on mpmath's experimental interval
arithmetic and the ideal unit-peak bump with fixed exact-binary inputs. It
does not establish a bound on a displaced $\gamma$ or $B$, omitted physical
processes, population tail, statistical closure, or total science
inference. Those terms remain separate in `predict` and may leave its total
budget `unbounded`. The envelope bounds stored basis coefficients; subsequent
floating-point matrix contractions are not separately interval-certified.
The coarse factor can be much wider than the actual angular coefficients,
particularly for derivative columns, large $(l,k)$ or deep Taylor rows.
The expensive certificate runs only in this explicit preparation call;
`predict` rebuilds the response matrix from the stored basis arrays for
reporting, while the fitting paths prepare their design matrix once. The
certified basis checks its matrix, numerical envelope and fixed-reference
inputs when either the matrix or a prediction budget is used. A modified
basis fails closed instead of carrying the old bound. Both prediction paths
propagate the stored column envelope without rerunning the certificate. A
parameter-varying model requires an envelope uniform over its declared
parameter domain. Manual consumers must obtain the numerical envelope through
`validated_numerical_term()`; directly reading the raw
`kernel_terms.numerical.value` field does not validate a modified basis.
Install `mpmath` for the optional certificate path.

For repeated spectra at a fixed basis, compile the moment-to-Stokes contraction
once. The following callable accepts a new moment vector on every invocation;
neither basis construction nor optional certification is in that loop:

```python
import jax
import jax.numpy as jnp
from syncmoments.model import JointMoments, predict

def spectra(moment_vector):
    moments = JointMoments.from_vector(
        bounded_basis.index, moment_vector, bounded_basis.reference
    )
    return predict(bounded_basis, moments, amplitude=1.0).stokes

spectra = jax.jit(spectra)
moment_vector = jnp.zeros(bounded_basis.index.n_real).at[0].set(1.0)
spectra(moment_vector).block_until_ready()  # compile before timing or inference
```

For an exploratory basis, substitute `basis` for `bounded_basis`; the resulting
numerical error status stays `unbounded`. A local one-channel, three-column
warm benchmark measured 3.90 microseconds for a compiled prepared-matrix
product and 4.58 microseconds for the compiled Stokes-only `predict` call
(100 invocations, median). Calling the eager `predict` path in the same
experiment took 2.40 milliseconds per call. These are local timings for a
fixed basis and moment-only inference, not a cost estimate for changing
$(\gamma,B)$ or rebuilding the basis.

In a local one-channel 100 MHz, 10%-half-width pilot at
$(\gamma,B)=(3000,5\,\mu\mathrm G)$, the exploratory 24-by-32 ridge basis
took 1.05 s to build. Three warmed offline certificate calls at
`i00_atol=1e-18` took 1.17, 0.99 and 0.98 s; the median was 0.99 s. The
fast $I_{00}$ was $7.0859104451\times10^{-22}$ and its attached absolute
numerical bound was $9.9854\times10^{-19}$, so this coarse example does not
establish a useful relative accuracy. The certificate retained 385 angular
cells and evaluated 119 dense mode blocks. These are timings and a
conditional numerical bound for one configuration, not a general speed or
scientific-adequacy claim.

In a separate 100 MHz, 20%-half-width `N=1`, depth-degree-zero test, the
8-by-12 fast basis took a fraction of a second to build. The optional
fixed-reference certificate with `i00_atol=2e-18` took about 1.57 s and
returned a finite maximum derivative-column error of about $9.0\times
10^{35}$ in the response's units. This is a valid *conditional coarse*
enclosure under the interval assumptions, but it is far too broad for a
useful science tolerance. A small `column_atol` therefore fails closed.
Obtaining a usable derivative-domain envelope remains open.

The result reports four separate numerical terms:

| Field | Status | Meaning |
| --- | --- | --- |
| `quadrature` | `estimate` for accelerated channels | Magnitude of the difference between the last two Gauss–Legendre rules, in rows $I,V,P_b$; it estimates the *integral evaluation* only. |
| `harmonic_sum` | `unbounded` for accelerated channels | The derivative remainder in DLMF 2.10.1 is not evaluated or bounded. |
| `bessel` | `unbounded` | The installed SciPy special-function evaluation has no package error certificate at these orders. |
| `floating_point` | `unbounded` | Frequency support, phase, and accumulation roundoff have no bound. |

For all-direct channels, `quadrature` and `harmonic_sum` are
`not_applicable`; Bessel and roundoff remain unbounded. The
`quadrature_converged` flag refers only to the difference of two rules. By
default a failed difference check raises `ArithmeticError`; an exploratory
result can be requested with `allow_unconverged=True` and retains a false
flag and unbounded harmonic-sum error.
Nonfinite response, phase, or accumulated channel values raise
`ArithmeticError`, including when a custom phase-weight function is supplied.
The public line evaluator also rejects an unrepresentable harmonic frequency,
including on its zero-emissivity branch.
The omitted physical processes, instrument response mismatch, and excluded
population tail are separate model uncertainties. Numerical overlap tests,
including a turning-region comparison, are finite evidence and cannot make
these terms rigorous bounds. The optional N<=1 basis certificate supplies a
conditional coefficient envelope to `predict` and fitting, but its
demonstrated high-$\gamma$ bounds are too wide to establish a useful science
tolerance. Narrow line-resolving channels are handled by the direct branch
when within `direct_limit`; other dense response families raise an error.

The large-order Bessel values come from SciPy's implementation. The code does
not truncate [DLMF 10.20.4](https://dlmf.nist.gov/10.20.E4) or
[10.20.7](https://dlmf.nist.gov/10.20.E7); citing these expansions here
identifies the future route for a documented uniform asymptotic evaluator,
not a claimed bound on the present implementation.
An internal fixed-point sparse route now has a separate integer-line Airy
enclosure based on the [periodic Bessel integral](https://dlmf.nist.gov/10.9.E2)
and the [Airy integral](https://dlmf.nist.gov/9.5.E1). It selects that route
only when its explicit line remainder meets the requested Bessel tolerance,
and otherwise uses a bounded contour quadrature. It also has a whole-range
magnitude bound for faint dense channels. These interval results rely on
mpmath's experimental interval arithmetic. The separate optional coarse
`N<=1` basis certificate above compares a whole-angular-domain enclosure to
the cached SciPy-based matrix; it does not certify the public pointwise
spectrum API in general.
An additional internal dense route splits the active integer interval into
blocks. A whole-block interval contains each finite-$\gamma$ line's weighted
$I,V,P_b$ term, including the Airy line remainder; multiplying by the number
of integers bounds the block sum. It bisects blocks until its componentwise
absolute error meets the requested scalar or per-channel tolerance, or
raises at a resource cap. This avoids a continuum summation assumption.
The route currently requires a unit-peak bump, the Airy turning region, a
fixed particle and angle, and optional Taylor phase. Its interval result is
conditional on the experimental mpmath interval arithmetic. The optional
whole-angular-domain route above uses these blocks only where eligible,
with sound whole-range magnitude fallbacks elsewhere; that pointwise route
does not itself bound parameter derivatives or change `build_basis`.
The fixed-point geometry evaluates the Doppler denominator and the small
`eta - beta*mu` factor through cancellation-resistant identities. A
high-precision near-axis check covers a case where rounding beta to one
would otherwise erase a nonzero Stokes V. These rearrangements do not
establish a floating-point error envelope.

For the fixed 0.1 GHz, 20%-half-width bump at
`(gamma, B, mu, eta) = (3000, 5e-6, 0, 0)`, a local warmed run took
about 0.0006 s on the public exploratory route. The internal bounded route
with `atol=1e-19` took about 0.33 s without a Faraday phase, retaining 117
blocks after 233 evaluations. With `TaylorPhase(0)` and `depth_ref=0.1`, its
measured median was about 0.56 s, retaining 194 blocks after 387
evaluations; the extra phase variation changes the subdivision work. These
routes have different numerical contracts and algorithms, so these timings
are workflow measurements rather than a like-for-like kernel benchmark.
The bounded route reuses Airy Taylor coefficients across blocks and includes
an explicit series truncation tail. Its ideal bump response hull uses the
response's monotonicity on the exact binary stored support, which tightens
intervals crossing a channel edge; host floating-point response error is
separate. Geometry is already computed once per
particle/angle point; a reusable table of Airy values as a function of its
turning argument would need an interpolation remainder before it could
preserve a bound. A table over frequency or particle parameters would also
need bounds for response, phase, and moving harmonic support.
