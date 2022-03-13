# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import numpy
import fractions
import math
import sympy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def ModPfy(func):
    @functools.wraps(func)
    def wrapper_ModPfy(self, other):
        if isinteger(other) or str(type(other)) == "long" or isinstance(other, fractions.Fraction):
            return func(self, ModP(other, self.p))
        elif isinstance(other, ModP):
            if self.p != other.p:
                raise ValueError("Numbers belong to different finite fields: FF{} and FF{}".format(self.p, other.p))
            return func(self, other)
        else:
            return NotImplemented
    return wrapper_ModPfy


def isinteger(x):
    return isinstance(x, int) or type(x) in [numpy.int32, numpy.int64]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class ModP(object):
    """Finite field with p elements ($\\mathbb{FF}_p$), i.e. integers modulus p, with p prime."""

    __slots__ = 'n', 'p'

    def __init__(self, n, p=None):
        from .padic import PAdic
        from .gaussian_rationals import GaussianRational
        if p is not None and isinteger(n) and isinteger(p):
            self.n = int(n) % int(p)
            self.p = int(p)
        elif p is not None and isinstance(n, fractions.Fraction):
            self_ = ModP(n.numerator, p) / ModP(n.denominator, p)
            self.n = self_.n
            self.p = self_.p
        elif p is not None and isinstance(n, GaussianRational):
            if n.imag != 0:
                raise ValueError("GaussianRational to ModP conversion requires zero part.")
            self_ = ModP(n.real.numerator, p) / ModP(n.real.denominator, p)
            self.n = self_.n
            self.p = self_.p
        elif p is None and isinstance(n, ModP):
            self.n = n.n
            self.p = n.p
        elif p is None and isinstance(n, PAdic):
            self.n = int(n)
            self.p = n.p ** n.k
        elif p is None:
            self.n, self.p = self.__rstr__(n)
        else:
            raise Exception('Bad finite field constructor, (n, p) of  value:({}, {}) and type:({}, {}).'.format(n, p, type(n), type(p)))

    def __getstate__(self):
        return (int(self), self.p)

    def __setstate__(self, state):
        self.__init__(*state)

    def __int__(self):
        return self.n

    def __abs__(self):
        return 0 if self.n == 0 else 1

    def __str__(self):
        return "%d %% %d" % (self, self.p)

    @staticmethod
    def __rstr__(string):
        return tuple(map(int, string.replace(" ", "").split("%")))

    def __repr__(self):
        return str(self)

    def __neg__(self):
        """Unary '-' operation"""
        return ModP(self.p - int(self), self.p)

    def __pos__(self):
        """Unary '+' operation"""
        return self

    @ModPfy
    def __eq__(self, other):
        return self.n == other.n and self.p == other.p

    @ModPfy
    def __add__(self, other):
        return ModP(int(self) + int(other), self.p)

    @ModPfy
    def __radd__(self, other):
        return ModP(int(other) + int(self), self.p)

    @ModPfy
    def __sub__(self, other):
        return ModP(int(self) - int(other), self.p)

    @ModPfy
    def __rsub__(self, other):
        return ModP(int(other) - int(self), self.p)

    @ModPfy
    def __mul__(self, other):
        return ModP(int(self) * int(other), self.p)

    @ModPfy
    def __rmul__(self, other):
        return ModP(int(other) * int(self), self.p)

    @ModPfy
    def __truediv__(self, other):
        if not isinstance(other, ModP):
            other = ModP(other, self.p)
        return self * other._inv()

    @ModPfy
    def __rtruediv__(self, other):
        return other * self._inv()

    def __pow__(self, other):
        assert(type(other) is int or hasattr(other, "is_integer") and other.is_integer() is True)
        if other > 0:
            return ModP(int(self) ** int(other), self.p)
        else:
            return 1 / ModP(int(self) ** - int(other), self.p)

    def _inv(self):
        """Find multiplicative inverse of self in Z_p (Z mod p) using the extended Euclidean algorithm."""

        s, t, gcd = extended_euclideal_algorithm(int(self), self.p)

        if gcd != 1:
            raise ZeroDivisionError("Inverse of {} mod {} does not exist. Are you sure {} is prime?".format(self, self.p, self.p))

        return ModP(s, self.p)

    def __hash__(self):
        return hash(self.n) + hash(self.p)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def extended_euclideal_algorithm(a, b):
    """Returns Bezout coefficients (s,t) and gcd(a,b) such that: as+bt=gcd(a,b). - Pseudocode from https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm"""
    (old_r, r) = (a, b)
    (old_s, s) = (1, 0)
    (old_t, t) = (0, 1)

    while r != 0:
        quotient = old_r // r
        (old_r, r) = (r, old_r - quotient * r)
        (old_s, s) = (s, old_s - quotient * s)
        (old_t, t) = (t, old_t - quotient * t)

    # output "Bézout coefficients:", (old_s, old_t)
    # output "greatest common divisor:", old_r
    # output "quotients by the gcd:", (t, s)

    return (old_s, old_t, old_r)


def gcd(a, b):
    _, _, gcd = extended_euclideal_algorithm(a, b)
    return gcd


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def MQRR(u, m, T=None):
    """Maximal Quotient Rational Reconstruction (M. B. Monagan)"""
    if T is None:
        c = 1
        T = 2 ** c * math.ceil(math.log(m, 2))
    if u == 0:
        if m > T:
            return 0
        else:
            return False
    (n, d) = (0, 0)
    (t0, r0) = (0, m)
    (t1, r1) = (1, u)
    while r1 != 0 and r0 > T:
        q = math.floor(r0 / r1)
        if q > T:
            (n, d, T) = (r1, t1, q)
        (r0, r1) = (r1, r0 - q * r1)
        (t0, t1) = (t1, t0 - q * t1)
    if d < 0:
        (n, d) = (-n, -d)
    if d == 0 or gcd(n, d) not in (1, d):
        return False
    return fractions.Fraction(n, d)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def LGreduction(u, v):
    """Lattice Gaussian Reduction - 2D version of LLL (Lenstra–Lenstra–Lovász)"""
    u, v = numpy.array(u, dtype=object), numpy.array(v, dtype=object)
    if v @ v > u @ u:
        return LGreduction(v, u)
    while v @ v < u @ u:
        (u, v) = (v, u)
        q = round(fractions.Fraction(u @ v, u @ u))
        v = v - q * u
    return (u, v)


def LGRR(a, n):
    """Lattice Gaussian Rational Reconstruction"""
    return fractions.Fraction(*LGreduction((a, 1), (n, 0))[0])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def rationalise(a, n=None, algorithm=(LGRR, MQRR)[1]):
    """Given (a, n) returns a fraction r / s such that r/s % n = a, by lattice reduction. r = sa + mn  <-> r/s % n = a"""
    if n is None:  # for FF argument
        if type(a) is int:
            return fractions.Fraction(a, 1)
        elif type(a) is ModP:
            return rationalise(int(a), a.p, algorithm)
    return algorithm(a, n)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def chinese_remainder(a1, a2):
    """Given a1 = a % n1 and a2 = a % n2 and assuming gcd(n1,n2)=1 (i.e. n1, n2 co-prime), returns a12 = a % (n1*n2)"""
    a1, n1 = int(a1), a1.p
    a2, n2 = int(a2), a2.p
    q1, q2, gcd = extended_euclideal_algorithm(n1, n2)
    assert gcd == 1
    return ModP(a1 * (q2 * n2) + a2 * (q1 * n1), n1 * n2)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def finite_field_sqrt(x):
    """Returns either False or the first digit of the root in the field."""
    from ..algebraic_geometry.tools import univariate_finite_field_solver
    from .field_extension import FieldExtension
    assert isinstance(x, ModP)
    root = univariate_finite_field_solver(f"x^2-{int(x)}", dict(), x.p)
    if root is False:
        return FieldExtension(x)
    else:
        return ModP(int(root[0][sympy.symbols('x')]), x.p)
