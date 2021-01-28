# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import random
import pytest

from lips.fields.finite_field import ModP, extended_euclideal_algorithm


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_picklable():
    obj = ModP('2 % 10007')
    mydump = pickle.dumps(obj, protocol=2)
    loaded = pickle.loads(mydump)
    assert obj == loaded


def test_addition():
    p = 10007
    a, b = random.randrange(0, 10000), random.randrange(0, 10000)
    assert (a + b) % p == ModP(a, p) + ModP(b, p)
    assert (a + b) % p == a + ModP(b, p)
    assert (a + b) % p == ModP(a, p) + b


def test_subtraction():
    p = 10007
    a, b = random.randrange(0, 10000), random.randrange(0, 10000)
    assert (a - b) % p == ModP(a, p) - ModP(b, p)
    assert (a - b) % p == a - ModP(b, p)
    assert (a - b) % p == ModP(a, p) - b


def test_multiplication():
    p = 10007
    a, b = random.randrange(0, 10000), random.randrange(0, 10000)
    assert (a * b) % p == ModP(a, p) * ModP(b, p)
    assert (a * b) % p == a * ModP(b, p)
    assert (a * b) % p == ModP(a, p) * b


def test_negation():
    p = 10007
    a = random.randrange(0, 10000)
    assert (-a) % p == -ModP(a, p)


def test_inverse():
    p = 10007
    a = ModP(random.randrange(1, p), p)
    assert a * a._inv() == 1


def test_failed_inverse():
    a = ModP('2 % 4')
    with pytest.raises(ZeroDivisionError):
        a * a._inv()


def test_extended_euclideal_algorithm():
    for i in range(10):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        s, t, gcd = extended_euclideal_algorithm(a, b)
        assert(a * s + b * t == gcd)
