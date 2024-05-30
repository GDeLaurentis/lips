# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import mpmath

from lips import Particles
from lips.tools import myException
from lips.invariants import Invariants

from tools import mapThreads, retry

mpmath.mp.dps = 300
UseParallelisation = True
Cores = 6

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.parametrize(
    "multiplicity, type1, type2, expected_failures, expected_length",
    [
        (5, "invs_2", "invs_2", 0, 380),
        (5, "invs_2", "invs_N", 24, 320),
        (5, "invs_3", "invs_3", 60, 90),
        (6, "invs_2", "invs_2", 0, 870),
        (6, "invs_2", "invs_N", 48, 4950),
        (6, "invs_2", "invs_s", 0, 300),
        (6, "invs_2", "invs_D", 0, 150),
        (6, "invs_3", "invs_3", 204, 8556),
        (6, "invs_3", "invs_s", 0, 930),
        (6, "invs_3", "invs_D", 117, 465),
        (6, "invs_s", "invs_s", 0, 90),
        (6, "invs_s", "invs_D", 20, 50),
    ]
)
def test_particles_set_pair(multiplicity, type1, type2, expected_failures, expected_length):

    oInvariants = Invariants(multiplicity, no_cached=True)
    tuples = [(inv1, inv2) for inv1 in getattr(oInvariants, type1) for inv2 in getattr(oInvariants, type2) if inv1 != inv2]

    TrueOrFalseList = mapThreads(DoubleScalingsTestingInner, multiplicity, oInvariants.full, tuples, UseParallelisation=UseParallelisation, Cores=Cores)
    failed_counter = sum(1 for entry in TrueOrFalseList if entry is False or entry is None)

    assert len(tuples) == expected_length
    assert failed_counter == expected_failures


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@retry((myException,), max_tries=2, silent=True)
def DoubleScalingsTestingInner(n, invariants, _tuple):
    oParticles = Particles(n)
    try:
        oParticles._set_pair(_tuple[0], 10 ** -28, _tuple[1], 10 ** -28)
        return True
    except Exception as e:
        print(e)
        return False
