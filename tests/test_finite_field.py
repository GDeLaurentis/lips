# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import random
import pytest

from fractions import Fraction
from lips.fields.finite_field import ModP, extended_euclideal_algorithm, rationalise, MQRR, LGRR


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


def test_failed_operation_different_FFs():
    a = ModP('2 % 3')
    b = ModP('2 % 5')
    with pytest.raises(ValueError):
        a + b


def test_addition_with_fraction_is_symmetric():
    a = ModP('2 % 5')
    b = Fraction(1, 3)
    assert a + b == b + a
    assert isinstance(a + b, ModP) and isinstance(b + a, ModP)


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
        assert a * s + b * t == gcd


def test_reconstruction_MQRR_2147483647():
    assert rationalise(298260199, 2147483647, algorithm=MQRR) == Fraction(-51071, 36)


def test_reconstruction_LGRR_2147483647():
    assert rationalise(298260199, 2147483647, algorithm=LGRR) == Fraction(11326, 42041)


def test_reconstruction_LGRR_2147483647_pow12():
    assert rationalise(4479489461410435237106627746985045825552416200368757674235908629372482248377113245126581341636163272702,
                       9619630365287747226839050681966839463919428531629782475127367001763589500187642982976435876178980851951693987841, algorithm=MQRR) == -1
    assert rationalise(4479489461410435237106627746985045825552416200368757674235908629372482248377113245126581341636163272702,
                       9619630365287747226839050681966839463919428531629782475127367001763589500187642982976435876178980851951693987841, algorithm=LGRR) == -1
