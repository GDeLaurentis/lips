# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy

from lips.gaussian_rationals import GaussianRational, rand_rat_frac


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_arithmetic_operations():

    a = GaussianRational(rand_rat_frac(), rand_rat_frac())
    b = GaussianRational(rand_rat_frac(), rand_rat_frac())

    assert(numpy.isclose(complex(a + b), complex(a) + complex(b)))
    assert(numpy.isclose(complex(a - b), complex(a) - complex(b)))
    assert(numpy.isclose(complex(a * b), complex(a) * complex(b)))
    assert(numpy.isclose(complex(a / b), complex(a) / complex(b)))
    assert(numpy.isclose(complex(a ** 17), complex(a) ** 17))
