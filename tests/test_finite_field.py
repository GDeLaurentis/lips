# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import random

from lips.fields.finite_field import ModP, extended_euclideal_algorithm


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_picklable():
    obj = ModP('2 % 10009')
    mydump = pickle.dumps(obj, protocol=2)
    loaded = pickle.loads(mydump)
    assert obj == loaded


def test_extended_euclideal_algorithm():
    for i in range(10):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        s, t, gcd = extended_euclideal_algorithm(a, b)
        assert(a * s + b * t == gcd)
