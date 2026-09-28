"""Interval-enclosed integer-order Bessel values on the nonoscillatory side.

For integer m, DLMF 10.9.2 gives

    J_m(m z) = (2 pi)^-1 int_0^(2 pi) exp(i m (t - z sin(t))) dt.

Periodicity permits shifting t to t+i*a. If 0<d<a, the integrand is
analytic on |Im(t)|<=d about the shifted contour. Put
phi(s)=s-z*sinh(s) and M=exp(-m*min(phi(a-d),phi(a+d))).
The Fourier coefficients of the shifted integrand obey
|c_k|<=M*exp(-|k|d), so the N-point periodic trapezoid has absolute error
at most 2*M/expm1(N*d). For the argument derivative J'_m(mz), the
integrand has an additional -i*sin(t+i*a), so its bound is multiplied by
cosh(a+d).

Every sampled integrand and the bound are evaluated with mpmath's interval
elementary functions. The enclosure is for the exact binary float or input
interval z supplied to this function, not for a separately rounded physical
geometry. mpmath's
interval support is marked experimental by its maintainers; this route is
kept internal until its elementary operations and performance have been
validated for the target environment. It does not address real noninteger
orders, harmonic summation, channel weights or the physical-model error.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class IntegerBesselEnclosure:
    """Intervals for J_m(mz) and its argument derivative over the input z."""

    j: Any
    j_prime: Any
    trapezoid_j: Any
    trapezoid_j_prime: Any
    nodes: int
    truncation_error: Any
    derivative_truncation_error: Any


def integer_bessel_enclosure(
    m: int,
    z: Any,
    *,
    abs_tolerance: float = 1e-10,
    max_nodes: int = 200_000,
    decimal_digits: int | None = None,
) -> IntegerBesselEnclosure:
    """Enclose one integer-order line by shifted-contour trapezoids.

    abs_tolerance controls the analytic quadrature remainder for both
    values. Interval rounding width is reported in the returned intervals
    and is not assumed to meet that tolerance. z may be a binary float or
    an mpmath interval. Only 0<=z<=1 is supported;
    the current shifted-contour branch accepts z>=0.5. Raises when the
    requested quadrature remainder exceeds the resource cap.
    """
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError("m must be a positive integer")
    if m > 2**53 - 2:
        raise ValueError("m exceeds the host channel's exact float-order limit")
    scalar_z = isinstance(z, (float, int)) and not isinstance(z, bool)
    if not scalar_z and not (hasattr(z, "a") and hasattr(z, "b")):
        raise ValueError("z must be a finite real scalar or interval in [0, 1]")
    if (
        isinstance(abs_tolerance, bool)
        or not math.isfinite(abs_tolerance)
        or abs_tolerance <= 0
    ):
        raise ValueError("abs_tolerance must be finite positive")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 4:
        raise ValueError("max_nodes must be an integer >= 4")
    if decimal_digits is None:
        decimal_digits = max(50, math.ceil(math.log10(m)) + 40)
    if (
        isinstance(decimal_digits, bool)
        or not isinstance(decimal_digits, int)
        or decimal_digits < 30
    ):
        raise ValueError("decimal_digits must be an integer >= 30")
    try:
        from mpmath.ctx_iv import MPIntervalContext  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("integer Bessel enclosures require mpmath") from exc

    ctx = MPIntervalContext()
    ctx.dps = decimal_digits
    zi = ctx.mpf(z) if scalar_z else ctx.mpf([z.a, z.b])
    if not (zi.a >= 0 and zi.b <= 1):
        raise ValueError("z must be a finite real scalar or interval in [0, 1]")
    if zi.a == 0 and zi.b == 0:
        zero = ctx.mpf(0)
        prime = ctx.mpf("0.5") if m == 1 else zero
        return IntegerBesselEnclosure(zero, prime, zero, prime, 0, zero, zero)
    if not zi.a >= 0.5:
        raise ValueError("shifted-contour branch requires z >= 0.5")

    # The contour height only affects cost and is not part of the proof.
    z_hint = float(zi.mid)
    a = math.acosh(1.0 / z_hint) if z_hint < 1 else m ** (-1.0 / 3.0)
    d = a / 2.0
    if not (math.isfinite(a) and a > 0 and d > 0):
        raise ValueError("a positive representable contour height is required")
    mi, ai, di = ctx.mpf(m), ctx.mpf(a), ctx.mpf(d)
    tolerance = ctx.mpf(abs_tolerance)

    def sinh(value):
        return (ctx.exp(value) - ctx.exp(-value)) / 2

    def cosh(value):
        return (ctx.exp(value) + ctx.exp(-value)) / 2

    def phi(value):
        return value - zi * sinh(value)

    # phi is concave for positive s. Its minimum on [a-d,a+d] is attained
    # at an endpoint. Use the lower endpoints of interval evaluations.
    minimum = min(phi(ai - di).a, phi(ai + di).a)
    majorant = ctx.exp(-mi * minimum).b
    derivative_factor = cosh(ai + di).b

    def remainders(nodes):
        denominator = (ctx.exp(nodes * di) - 1).a
        base = (2 * majorant / denominator).b
        derivative = (base * derivative_factor).b
        return base, derivative

    lower, upper = 4, max_nodes
    _, derivative_error = remainders(upper)
    if not derivative_error <= tolerance:
        raise ValueError("requested Bessel quadrature tolerance exceeds max_nodes")
    while lower < upper:
        middle = (lower + upper) // 2
        _, derivative_error = remainders(middle)
        if derivative_error <= tolerance:
            upper = middle
        else:
            lower = middle + 1
    nodes = lower
    value_error, derivative_error = remainders(nodes)

    total = ctx.mpf(0)
    derivative_total = ctx.mpf(0)
    # For integer m and real z, samples k and N-k are conjugate on this
    # contour. The k=0 and (for even N) k=N/2 samples are self-paired.
    for k in range(nodes // 2 + 1):
        theta = 2 * ctx.pi * k / nodes + 1j * ai
        sine = ctx.sin(theta)
        term = ctx.exp(1j * mi * (theta - zi * sine))
        multiplicity = 1 if k == 0 or 2 * k == nodes else 2
        total += multiplicity * term.real
        derivative_total += multiplicity * (-1j * sine * term).real
    symmetric_error = ctx.mpf([-1, 1])
    trapezoid_value = total / nodes
    trapezoid_derivative = derivative_total / nodes
    value = trapezoid_value + symmetric_error * value_error
    derivative = trapezoid_derivative + symmetric_error * derivative_error
    return IntegerBesselEnclosure(
        value,
        derivative,
        trapezoid_value,
        trapezoid_derivative,
        nodes,
        value_error,
        derivative_error,
    )


__all__ = ["IntegerBesselEnclosure", "integer_bessel_enclosure"]
