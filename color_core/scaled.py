"""Float64 mantissas with integer binary exponents, not arbitrary precision reals.

Used where the specified equations exceed IEEE's exponent range. Addition rounds
at the mantissa; tiny pole margins are carried separately from their 400 baseline.
"""
import numpy as np


class Scalar:
    __slots__ = ('m', 'e')

    def __init__(self, value=0., exponent=0):
        if isinstance(value, Scalar):
            self.m, self.e = value.m, value.e
            return
        if isinstance(value, int) and abs(value).bit_length() > 1023:
            shift = abs(value).bit_length()-53
            leading, remainder = divmod(abs(value), 1 << shift)
            halfway = 1 << (shift-1)
            leading += remainder > halfway or (remainder == halfway and leading % 2)
            exponent += shift
            value = -leading if value < 0 else leading
        value = float(value)
        if not np.isfinite(value):
            raise ValueError('Scaled scalars require finite mantissas')
        self.m, shift = np.frexp(value)
        self.e = int(exponent)+int(shift) if self.m else 0

    def __float__(self):
        if not self.m or self.e < -1074:
            return float(np.copysign(0., self.m))
        if self.e > 1024:
            raise OverflowError('Scaled value is outside float64')
        with np.errstate(over='ignore', under='ignore'):
            value = float(np.ldexp(self.m, self.e))
        if not np.isfinite(value):
            raise OverflowError('Scaled value is outside float64')
        return value

    def __bool__(self):
        return bool(self.m)

    def __neg__(self):
        return Scalar(-self.m, self.e)

    def __abs__(self):
        return Scalar(abs(self.m), self.e)

    def __add__(self, other):
        other = Scalar(other)
        if not self:
            return other
        if not other:
            return self
        high, low = (self, other) if self.e >= other.e else (other, self)
        gap = low.e-high.e
        return Scalar(high.m+(np.ldexp(low.m, gap) if gap >= -1074 else 0.), high.e)

    __radd__ = __add__

    def __sub__(self, other):
        return self+-Scalar(other)

    def __rsub__(self, other):
        return Scalar(other)+-self

    def __mul__(self, other):
        other = Scalar(other)
        return Scalar(self.m*other.m, self.e+other.e)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Scalar(other)
        if not other:
            raise ZeroDivisionError('Scaled division by zero')
        return Scalar(self.m/other.m, self.e-other.e)

    def __rtruediv__(self, other):
        return Scalar(other)/self

    def __eq__(self, other):
        other = Scalar(other)
        return self.m == other.m and self.e == other.e

    def __lt__(self, other):
        other = Scalar(other)
        if np.sign(self.m) != np.sign(other.m):
            return self.m < other.m
        if not self:
            return False
        return (self.e, self.m) < (other.e, other.m) if self.m > 0 else (-self.e, self.m) < (-other.e, other.m)

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return Scalar(other) < self

    def __ge__(self, other):
        return Scalar(other) <= self

    def __pow__(self, power):
        if not self:
            if power < 0:
                raise ZeroDivisionError('Negative power at zero')
            return Scalar(1. if power == 0 else 0.)
        if self.m < 0:
            if int(power) != power:
                raise ValueError('Fractional power requires a nonnegative scalar')
            return (-1 if int(power) % 2 else 1)*(abs(self)**power)
        numerator, denominator = float(power).as_integer_ratio()
        integer, remainder = divmod(self.e*numerator, denominator)
        return Scalar(np.exp2(np.log2(self.m)*power+remainder/denominator), integer)

    def floor(self):
        numerator, denominator = float(self.m).as_integer_ratio()
        if self.e >= 0:
            return (numerator << self.e)//denominator
        # Huge negative exponents must not allocate a huge integer denominator.
        if self.e < -1074:
            return -1 if self.m < 0 else 0
        return numerator//(denominator << -self.e)

    def exp2(self):
        integer = self.floor()
        # Subtraction keeps the fractional part small even for a huge integer.
        if self.e > 53:
            fraction = 0.  # Every float64 mantissa at this scale is an integer.
        elif self.e < -1074:
            fraction = 1. if self.m < 0 else 0.
        else:
            fraction = float(self)-integer
        return Scalar(np.exp2(fraction), integer)

    def modulo(self, period):
        numerator, denominator = float(self.m).as_integer_ratio()
        if self.e >= 0:
            return ((numerator << self.e) % (int(period)*denominator))/denominator
        if self.e < -1074:
            return 0.
        denominator <<= -self.e
        return (numerator % (int(period)*denominator))/denominator


def total(values):
    """Cancel terms of a common scale before rounding away smaller terms."""
    values = [Scalar(v) for v in values if v]
    groups = {}
    for v in values:
        groups.setdefault(v.e, []).append(v.m)
    result = Scalar()
    for exponent in sorted(groups, reverse=True):
        result += Scalar(np.sum(groups[exponent], dtype=np.float64), exponent)
    return result


def dot(a, b):
    return total(Scalar(x)*y for x, y in zip(a, b))


def norm(values):
    return total(Scalar(v)**2 for v in values)**.5


def signed_power(value, power):
    """Sign-preserving real power, for the model's positive exponents."""
    if isinstance(value, Scalar):
        return (-1 if value < 0 else 1)*abs(value)**power
    value = np.asarray(value, dtype=np.float64)
    return np.sign(value)*np.abs(value)**power


def smooth_floor(value, lower):
    x = Scalar(value)-lower
    if x <= 0:
        return Scalar(lower)
    if x >= 1:
        return Scalar(value)
    return Scalar(lower)+x**3*(6+x*(-8+3*x))


def smooth_cap(value, upper):
    value, upper = Scalar(value), Scalar(upper)
    if value <= upper-1:
        return value
    if value >= upper:
        return upper
    x = upper-value
    return upper-x**3*(6+x*(-8+3*x))
