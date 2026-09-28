"""Conditional interval Airy enclosure for integer Bessel turning-point lines.

For integer m, DLMF 10.9.2 gives the periodic integral of J_m(mz). Shift
its contour to t+i*a, split at |t|=T<pi, replace sin(t+i*a) by its cubic
Taylor polynomial on the inner interval, and bound both outer tails. The
resulting infinite cubic integral is a scaled Airy Ai integral (DLMF 9.5.1).
Taylor's complex remainder, Gaussian moments, and explicit tail estimates
bound the difference for J_m and its argument derivative. Ai and Ai' are
enclosed by a power series with a geometric majorant for the omitted terms.

The enclosure uses mpmath's experimental interval elementary functions and
exact binary-float inputs. It is internal until that arithmetic is vetted.
This is an integer-line result; it does not bound continuous-order SciPy
Bessel values or a dense weighted harmonic sum.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class IntegerAiryEnclosure:
    j: Any
    j_prime: Any
    airy_j: Any
    airy_j_prime: Any
    j_error: Any
    j_prime_error: Any
    airy_series_error: Any
    airy_derivative_series_error: Any


def _magnitude(value):
    return max(abs(value.a), abs(value.b))


def _airy_coefficients(ctx, terms=180):
    """Build interval Ai Taylor coefficients once for one precision context."""
    third = ctx.mpf(1) / 3
    a0 = 1 / (ctx.exp(2 * third * ctx.ln(3)) * ctx.gamma(2 * third))
    a1 = -1 / (ctx.exp(third * ctx.ln(3)) * ctx.gamma(third))
    coefficients = [a0, a1, ctx.mpf(0)]
    for n in range(1, terms + 1):
        coefficients.append(coefficients[n - 1] / ((n + 2) * (n + 1)))
    return coefficients


def _airy_series(ctx, x, *, terms=None, coefficients=None):
    """Enclose Ai(x), Ai'(x) for 0<=x<=10 by y''=x*y.

    By default, choose a truncation whose explicit geometric tails are below
    the interval context's working precision, up to 180 terms.
    The returned tails remain part of the enclosure at any truncation.
    """
    xmax = _magnitude(x)
    if not (x.a >= 0 and xmax <= 10):
        raise ValueError("Airy series requires 0 <= x <= 10")
    if terms is not None and (
        isinstance(terms, bool) or not isinstance(terms, int) or terms < 10
    ):
        raise ValueError("Airy series terms must be an integer >= 10")
    if coefficients is None:
        coefficients = _airy_coefficients(ctx, 180 if terms is None else terms)

    def tails(count):
        first = range(count, count + 3)
        ratio = (xmax**3 / ((count + 3) * (count + 2))).b
        derivative_ratio = (xmax**3 / (count * (count + 2))).b
        if not (ratio < 1 and derivative_ratio < 1):
            return None
        tail = sum(_magnitude(coefficients[n]) * xmax**n for n in first) / (1 - ratio)
        derivative_tail = sum(
            n * _magnitude(coefficients[n]) * xmax ** (n - 1) for n in first
        ) / (1 - derivative_ratio)
        return tail.b, derivative_tail.b

    if terms is None:
        target = (ctx.mpf(10) ** (2 - ctx.dps)).a
        # This initial guess only saves work: every accepted truncation is
        # checked against the outward interval tail before use.
        first_candidate = min(
            180, max(20, 10 * math.ceil((ctx.dps - 20 + 20 * float(xmax)) / 10))
        )
        for candidate in range(first_candidate, 181, 10):
            selected_tails = tails(candidate)
            if selected_tails is not None and max(selected_tails) <= target:
                terms = candidate
                break
        else:
            terms = 180
    else:
        selected_tails = tails(terms)
    if selected_tails is None:
        raise ValueError("Airy series tail ratio is not below one")
    tail, derivative_tail = selected_tails
    # The Airy recurrence makes every coefficient with n = 2 (mod 3) zero.
    # Evaluate P(x) = A(x^3) + x B(x^3) and its derivative by Horner. This
    # keeps every operation interval rounded and uses only two shorter series.
    cube = x**3
    even_value = even_derivative = ctx.mpf(0)
    odd_value = odd_derivative = ctx.mpf(0)
    for k in range((terms - 1) // 3, -1, -1):
        even_derivative = even_derivative * cube + even_value
        even_value = even_value * cube + coefficients[3 * k]
        odd_derivative = odd_derivative * cube + odd_value
        odd_value = odd_value * cube + (
            coefficients[3 * k + 1] if 3 * k + 1 < terms else 0
        )
    value = even_value + x * odd_value
    derivative = 3 * x**2 * even_derivative + odd_value + 3 * cube * odd_derivative
    return (
        value + ctx.mpf([-tail, tail]),
        derivative + ctx.mpf([-derivative_tail, derivative_tail]),
        tail,
        derivative_tail,
    )


def _airy_on_interval(ctx, x, *, coefficients=None, endpoint_cache=None):
    """Tight Ai/Ai' hull using monotonicity on nonnegative real arguments."""
    if not (x.a >= 0 and x.b <= 10):
        raise ValueError("Airy interval requires 0 <= x <= 10")

    def evaluate(endpoint):
        point = ctx.mpf(endpoint)
        if endpoint_cache is None:
            return _airy_series(ctx, point, coefficients=coefficients)[:2]
        # _mpi_ is the exact binary interval representation. Decimal string
        # keys could alias distinct outward-rounded endpoints.
        key = point._mpi_
        if key not in endpoint_cache:
            endpoint_cache[key] = _airy_series(ctx, point, coefficients=coefficients)[
                :2
            ]
        return endpoint_cache[key]

    if x.a == x.b:
        return evaluate(x.a)
    low_ai, low_prime = evaluate(x.a)
    high_ai, high_prime = evaluate(x.b)
    # DLMF 9.9: Ai and Ai' have no nonnegative zeros; DLMF 9.2 gives
    # Ai(0)>0, Ai'(0)<0 and Ai''=x*Ai. Thus Ai decreases and Ai' increases.
    # https://dlmf.nist.gov/9.9 and https://dlmf.nist.gov/9.2.E1
    return (
        ctx.mpf([high_ai.a, low_ai.b]),
        ctx.mpf([low_prime.a, high_prime.b]),
    )


def _contour_errors(ctx, mi, zi, a, T, *, gaussian_gamma=None):
    """Uniform cubic-replacement errors for all integer orders in ``mi``."""
    zlo, zhi = zi.a, zi.b
    sinh_a = (ctx.exp(a) - ctx.exp(-a)) / 2
    cosh_a = (ctx.exp(a) + ctx.exp(-a)) / 2
    E_true = ctx.exp(-mi * (a - zhi * sinh_a)).b
    E_cubic = ctx.exp(-mi * (a * (1 - zhi) - zhi * a**3 / 6)).b
    E_inner = max(E_true, E_cubic)
    inner_ratio = (2 * ctx.sin(T / 2) ** 2 / T**2).a
    c_true = (mi * zlo * sinh_a * inner_ratio).a
    c_cubic = (mi * zlo * a / 2).a
    c_inner = min(c_true, c_cubic)
    c_tail = (2 * mi * zlo * sinh_a / ctx.pi**2).a
    if not (c_inner > 0 and c_tail > 0):
        raise ArithmeticError("Airy contour decay bound must be positive")

    if gaussian_gamma is None:
        gaussian_gamma = tuple(ctx.gamma(ctx.mpf(k + 1) / 2) for k in range(7))

    def gaussian_moment(power, decay):
        # Integral over the real line of (|t|+a)^power exp(-decay*t^2).
        root = ctx.sqrt(decay)
        result = ctx.mpf(0)
        for k in range(power + 1):
            denominator = decay ** ((k + 1) // 2)
            if k % 2 == 0:
                denominator *= root
            result += (
                math.comb(power, k) * a ** (power - k) * gaussian_gamma[k] / denominator
            )
        return result.b

    inner_j = (
        mi * zhi * cosh_a * E_inner * gaussian_moment(5, c_inner) / (120 * 2 * ctx.pi)
    ).b
    inner_prime = (
        cosh_a * E_true * gaussian_moment(3, c_true) / (6 * 2 * ctx.pi)
        + mi * zhi * cosh_a * E_inner * gaussian_moment(6, c_inner) / (120 * 2 * ctx.pi)
    ).b
    true_tail = (E_true * ctx.exp(-c_tail * T**2)).b
    cubic_tail = (E_cubic * ctx.exp(-c_cubic * T**2) / (2 * ctx.pi * c_cubic * T)).b
    cubic_prime_tail = (
        E_cubic * ctx.exp(-c_cubic * T**2) * (1 + a / T) / (2 * ctx.pi * c_cubic)
    ).b
    error_j = (inner_j + true_tail + cubic_tail).b
    error_prime = (inner_prime + cosh_a * true_tail + cubic_prime_tail).b
    return error_j, error_prime


def integer_bessel_airy_enclosure(
    m: int,
    z: Any,
    *,
    decimal_digits: int = 70,
) -> IntegerAiryEnclosure:
    """Enclose J_m(mz), J'_m(mz) in a restricted turning-point region.

    The elementary Gaussian estimates are valid more widely, but this
    implementation requires m>=1000, 0.9<=z<=1, Airy argument <=10 and
    T=10*(2/(m*z_mid))^(1/3)<pi to keep the stated bound useful.
    """
    if isinstance(m, bool) or not isinstance(m, int) or not 1000 <= m <= 2**53 - 2:
        raise ValueError("m must be an exact integer in [1000, 2^53-2]")
    if (
        isinstance(decimal_digits, bool)
        or not isinstance(decimal_digits, int)
        or decimal_digits < 50
    ):
        raise ValueError("decimal_digits must be an integer >= 50")
    try:
        from mpmath.ctx_iv import MPIntervalContext  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("integer Airy enclosures require mpmath") from exc
    ctx = MPIntervalContext()
    ctx.dps = decimal_digits
    scalar_z = isinstance(z, (float, int)) and not isinstance(z, bool)
    if not scalar_z and not (hasattr(z, "a") and hasattr(z, "b")):
        raise ValueError("z must be a scalar or interval")
    zi = ctx.mpf(z) if scalar_z else ctx.mpf([z.a, z.b])
    if not (zi.a >= ctx.mpf("0.9") and zi.b <= 1):
        raise ValueError("Airy enclosure requires 0.9 <= z <= 1")
    mi = ctx.mpf(m)
    scale = ctx.exp(ctx.ln(2 / (mi * zi)) / 3)
    a = ctx.mpf((2 / (m * float(zi.mid))) ** (1 / 3))
    T = 10 * a
    if not T.b < ctx.pi.a:
        raise ValueError("Airy contour split must lie below pi")
    x = mi * (1 - zi) * scale
    coefficients = _airy_coefficients(ctx)
    ai, ai_prime = _airy_on_interval(ctx, x, coefficients=coefficients)
    _, _, series_error, derivative_series_error = _airy_series(
        ctx, ctx.mpf(x.b), coefficients=coefficients
    )
    airy_j = scale * ai
    airy_j_prime = -(scale**2) * ai_prime

    error_j, error_prime = _contour_errors(ctx, mi, zi, a, T)
    return IntegerAiryEnclosure(
        airy_j + ctx.mpf([-error_j, error_j]),
        airy_j_prime + ctx.mpf([-error_prime, error_prime]),
        airy_j,
        airy_j_prime,
        error_j,
        error_prime,
        series_error,
        derivative_series_error,
    )


__all__ = [
    "IntegerAiryEnclosure",
    "integer_bessel_airy_enclosure",
]
