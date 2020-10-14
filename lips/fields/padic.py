# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import numpy

from .finite_field import ModP


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def to_base(num, p):
    if num < p:
        return (num, )
    else:
        return (num % p, ) + (to_base(num // p, p))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def padicfy(func):
    @functools.wraps(func)
    def wrapper_padicfy(self, other):
        if type(other) is PAdic:
            return func(self, other)
        elif type(other) in [int, ModP, numpy.int64] or str(type(other)) == "long":
            return func(self, PAdic(other, self.p, (self.k + self.n) if (self.k + self.n) > 0 else 0))
        else:
            return NotImplemented
    return wrapper_padicfy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class PAdic(object):

    """PAdic Integers, with p prime."""

    def __init__(self, num, p=None, k=None, n=0, from_addition=False, recover_precision_from_powers_of_p=True):
        """0 ≤ num ≤ p ^ k; p: prime; k: significant digits; n: power of prefactors of p."""
        if p is not None and k is not None:
            factors_of_p = next((i for i, j in enumerate(to_base(num, p)) if j != 0), 0)
            self.num = int(num // p ** factors_of_p) % int(p ** k)
            self.p = int(p)
            if from_addition is False or (recover_precision_from_powers_of_p and self.num == 1 and factors_of_p > 0):
                if from_addition is True:
                    print("!Warning! recovering a digit.")
                self.k = int(k)
            else:
                self.k = int(k) - from_addition * factors_of_p
            self.n = int(factors_of_p) + n
        elif p is None and k is None:
            self.num = num
        else:
            raise Exception("Invalid p-adic initialisation")

    def __int__(self):
        return self.num

    def __getstate__(self):
        return (int(self), self.p, self.k, self.n)

    def __setstate__(self, state):
        self.__init__(*state)

    def __abs__(self):
        return self.p ** -self.n

    @padicfy
    def __eq__(self, other):
        return all([int(self) == int(other), self.p == other.p, self.k == other.k, self.n == other.n])

    @property
    def as_tuple(self):
        return (to_base(int(self), self.p) + tuple([0 for i in range(self.k)]))[:self.k]

    def __str__(self):
        if self == 0:
            return "O({}^{})".format(self.p, self.n + self.k)
        else:
            return " + ".join(filter(lambda x: x is not None,
                                     ["{}".format(i) if (j == 0 and i != 0) else
                                      "{}*{}".format(i, self.p) if (j == 1 and i != 0) else
                                      "{}*{}^{}".format(i, self.p, j) if (i != 0) else None
                                      for i, j in zip(self.as_tuple, range(self.n, self.n + self.k))])) + " + O({}^{})".format(self.p, self.n + self.k)

    def __repr__(self):
        return str(self)

    @padicfy
    def __add__(self, other):
        if self.n > other.n:
            return other + self
        else:
            return PAdic((int(self) + int(other) * self.p ** (other.n - self.n)) % self.p ** self.k, self.p,  # min([self.k, other.k]),
                         ((self.k + self.n) if (self.k + self.n) < (other.k + other.n) else (other.k + other.n)) - self.n, self.n, from_addition=True)

    @padicfy
    def __radd__(self, other):
        return other + self

    @padicfy
    def __sub__(self, other):
        return self + (- other)

    @padicfy
    def __rsub__(self, other):
        return - (self - other)

    @padicfy
    def __mul__(self, other):
        return PAdic((int(self) * int(other)) % self.p ** self.k, self.p, min([self.k, other.k]), self.n + other.n)

    @padicfy
    def __rmul__(self, other):
        return self * other

    @padicfy
    def __truediv__(self, other):
        return PAdic((int(self) * ModP(int(other), other.p ** other.k)._inv()) % self.p ** self.k, self.p, min([self.k, other.k]), self.n - other.n)

    @padicfy
    def __div__(self, other):
        return self.__truediv__(other)

    @padicfy
    def __rtruediv__(self, other):
        return other / self

    @padicfy
    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __neg__(self):
        return PAdic((-1 * int(self)) % self.p ** self.k, self.p, self.k, self.n)

    def __pow__(self, n):
        assert(isinstance(n, int) or n.is_integer())
        if n == 0:
            return PAdic(1, self.p, self.k, 0)
        elif n % 2 == 0:
            root_2_res = self ** (n / 2)
            return root_2_res * root_2_res
        else:
            return self * (self ** (n - 1))
