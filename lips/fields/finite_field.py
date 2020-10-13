# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def TypeErrorCheck(func):
    @functools.wraps(func)
    def wrapper_TypeErrorCheck(self, other):
        try:
            return func(self, other)
        except TypeError:
            return NotImplemented
    return wrapper_TypeErrorCheck


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class ModP(int):
    'Integers modulus p, with p prime.'

    # __slots__ = 'p'

    def __new__(cls, *args, **kwargs):
        if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):  # usually this should get called
            return int.__new__(cls, args[0] % args[1])
        elif len(args) == 1 and isinstance(args[0], int):  # this is needed for pickling
            return int.__new__(cls, args[0])
        elif len(args) == 1:
            return int.__new__(cls, cls.__rstr__(args[0])[0])
        else:
            raise Exception('Bad finite field constructor.')

    def __init__(self, *args, **kwargs):
        if len(args) == 2:
            self.p = args[1]
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

    @TypeErrorCheck
    def __add__(self, other):
        return ModP(int(self) + int(other), self.p)

    @TypeErrorCheck
    def __radd__(self, other):
        return ModP(int(other) + int(self), self.p)

    @TypeErrorCheck
    def __sub__(self, other):
        return ModP(int(self) - int(other), self.p)

    @TypeErrorCheck
    def __rsub__(self, other):
        return ModP(int(other) - int(self), self.p)

    @TypeErrorCheck
    def __mul__(self, other):
        return ModP(int(self) * int(other), self.p)

    @TypeErrorCheck
    def __rmul__(self, other):
        return ModP(int(other) * int(self), self.p)

    @TypeErrorCheck
    def __truediv__(self, other):
        if not isinstance(other, ModP):
            other = ModP(other, self.p)
        return self * other._inv()

    @TypeErrorCheck
    def __rtruediv__(self, other):
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
            raise ValueError("Inverse of {} mod {} does not exist. Are you sure {} is prime?".format(self, self.p, self.p))

        return ModP(tcurr, self.p)
