# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import numpy
import fractions


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
        if p is not None and isinteger(n) and isinteger(p):
            self.n = int(n) % int(p)
            self.p = int(p)
        elif p is not None and isinstance(n, fractions.Fraction):
            self_ = ModP(n.numerator, p) / ModP(n.denominator, p)
            self.n = self_.n
            self.p = self_.p
        elif p is None and isinstance(n, ModP):
            self.n = n.n
            self.p = n.p
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
        return ModP(self.p - int(self), self.p)

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
        assert(type(other) is int)
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def rationalise(a, n=None):
    """Given (a, n) returns a fraction r / s such that r/s % n = a, by lattice reduction. r = sa + mn  <-> r/s % n = a"""
    if n is None:  # for FF argument
        if type(a) is int:
            return fractions.Fraction(a, 1)
        elif type(a) is ModP:
            return rationalise(int(a), a.p)
    return fractions.Fraction(*LGreduction((a, 1), (n, 0))[0])


def LGreduction(u, v):
    u, v = numpy.array(u, dtype=object), numpy.array(v, dtype=object)
    if v @ v > u @ u:
        return LGreduction(v, u)
    while v @ v < u @ u:
        (u, v) = (v, u)
        q = round(fractions.Fraction(u @ v, u @ u))
        v = v - q * u
    return (u, v)


def chinese_remainder(a1, a2):
    """Given a1 = a % n1 and a2 = a % n2 and assuming gcd(n1,n2)=1 (i.e. n1, n2 co-prime), returns a12 = a % (n1*n2)"""
    a1, n1 = int(a1), a1.p
    a2, n2 = int(a2), a2.p
    q1, q2, gcd = extended_euclideal_algorithm(n1, n2)
    assert gcd == 1
    return ModP(a1 * (q2 * n2) + a2 * (q1 * n1), n1 * n2)
