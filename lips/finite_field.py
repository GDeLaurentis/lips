#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class ModP(int):
    'Integers modulus p, with p prime.'

    def __new__(cls, num, p):
        assert type(num) in [int, long] and type(p) in [int, long], "Non integer modulus."
        self = int.__new__(cls, int(num) % int(p))
        self.p = int(p)
        return self

    def __str__(self):
        return "%d %% %d" % (self, self.p)

    def __repr__(self):
        return str(self)

    def __neg__(self):
        return ModP(self.p - int(self), self.p)

    def __add__(self, other):
        return ModP(int(self) + int(other), self.p)

    def __radd__(self, other):
        return ModP(int(other) + int(self), self.p)

    def __sub__(self, other):
        return ModP(int(self) - int(other), self.p)

    def __rsub__(self, other):
        return ModP(int(other) - int(self), self.p)

    def __mul__(self, other):
        return ModP(int(self) * int(other), self.p)

    def __rmul__(self, other):
        return ModP(int(other) * int(self), self.p)

    def __div__(self, other):
        if not isinstance(other, ModP):
            other = ModP(other, self.p)
        return self * other._inv()

    def __rdiv__(self, other):
        return other * self._inv()

    def __pow__(self, other):
        assert(type(other) is int)
        return ModP(int(self) ** int(other), self.p)

    def _inv(self):
        'Find multiplicative inverse of self in Z mod p using the extended Euclidean algorithm.'

        rcurr = self.p
        rnext = int(self)
        tcurr = 0
        tnext = 1

        while rnext:
            q = rcurr // rnext
            rcurr, rnext = rnext, rcurr - q * rnext
            tcurr, tnext = tnext, tcurr - q * tnext

        if rcurr != 1:
            raise ValueError("Inverse of {} mod {} does not exist. Are you sure %d is prime?" % (self, self.p, self.p))

        return ModP(tcurr, self.p)
