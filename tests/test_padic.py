# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import random

from lips.fields.padic import PAdic


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_picklable():
    obj = PAdic(3 + 2 * 7, 7, 3)
    mydump = pickle.dumps(obj, protocol=2)
    loaded = pickle.loads(mydump)
    assert obj == loaded


def test_str_non_zero():
    p, k = 10007, 3
    assert str(PAdic(1 + 2 * p + 3 * p ** 2, p, k)) == "1 + 2*{p} + 3*{p}^2 + O({p}^{k})".format(p=p, k=k)


def test_addition():
    p, k = 10007, 3
    a, b = random.randrange(0, 10000), random.randrange(0, 10000)
    assert PAdic(a + b, p, k) == PAdic(a, p, k) + PAdic(b, p, k)
    assert PAdic(a + b, p, k) == a + PAdic(b, p, k)
    assert PAdic(a + b, p, k) == PAdic(a, p, k) + b


def test_addition_with_zero():
    p, k = 10007, 3
    a = PAdic(random.randrange(0, 10000), p, k)
    assert a + 0 == a
