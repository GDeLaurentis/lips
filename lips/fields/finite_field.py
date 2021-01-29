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
        if type(other) in [int, ModP, numpy.int64] or str(type(other)) == "long":
            return func(self, ModP(other, self.p))
        elif type(other) is fractions.Fraction:
            return func(self, ModP(other.numerator, self.p) / ModP(other.denominator, self.p))
        else:
            return NotImplemented
    return wrapper_ModPfy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class ModP(int):
    'Integers modulus p, with p prime.'

    # __slots__ = 'p'

    def __new__(cls, *args, **kwargs):
        from .padic import PAdic
        if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):  # usually this should get called
            return int.__new__(cls, args[0] % args[1])
        elif len(args) == 1 and isinstance(args[0], int):  # this is needed for pickling
            return int.__new__(cls, args[0])
        elif len(args) == 1 and isinstance(args[0], PAdic):
            return int.__new__(cls, int(args[0]))
        elif len(args) == 1:
            return int.__new__(cls, cls.__rstr__(args[0])[0])
        else:
            raise Exception('Bad finite field constructor. args:{} of type:{}, kwargs:{} of type:{}.'.format(args, list(map(type, args)), kwargs, list(map(type, kwargs))))

    def __init__(self, *args, **kwargs):
        from .padic import PAdic
        if len(args) == 2:
            self.p = args[1]
        elif len(args) == 1 and isinstance(args[0], PAdic):
            self.p = args[0].p ** args[0].k
        elif len(args) == 1:
            self.p = self.__rstr__(args[0])[1]
        else:
            raise Exception('Bad finite field constructor.')

    def __getstate__(self):
        return (int(self), self.p)

    def __setstate__(self, state):
        self.__init__(*state)

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
