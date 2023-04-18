# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sympy
import pytest
import mpmath

from lips import Particles, Field
from lips.invariants import Invariants

from tools import mapThreads

mpmath.mp.dps = 300
UseParallelisation = True
Cores = 6

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.parametrize("multiplicity", range(4, 8))
def test_particles_set(multiplicity):
    invariants = Invariants(multiplicity, no_cached=True).full
    TrueOrFalseList = mapThreads(SingleScalingsTestingInner, multiplicity, invariants, invariants, UseParallelisation=UseParallelisation, Cores=Cores)
    failed_counter = sum(1 for entry in TrueOrFalseList if entry is False)
    assert failed_counter == 0
    # print("\r{}/{} single collinear limits failed to be constructed.                                  \n".format(failed_counter, len(invariants)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def SingleScalingsTestingInner(n, invariants, invariant):
    max_error_nbr = 1
    error_counter = 0
    while True:
        oParticles = Particles(n)
        result = oParticles._set(invariant, 10 ** -28)
        _, _, _, small_invs = oParticles.phasespace_consistency_check(invariants)
        if len(small_invs) > 1:         # check that only one is set to be small
            error_counter += 1
            sympy.pprint(result)
            sympy.pprint(invariant)
            sympy.pprint(small_invs)
            if error_counter == max_error_nbr + 1:
                return False
        elif result is False:           # if it fails print it
            error_counter += 1
            print("\r", result, invariant)
            if error_counter == max_error_nbr + 1:
                return False
        else:
            return True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_particles_set_p5Bdiff():
    invariants = ['(⟨2|3|5+6|1|2]-⟨3|4|5+6|1|3])', '(⟨3|2|5+6|4|3]-⟨2|1|5+6|4|2])']
    oPs = Particles(6, field=Field("finite field", 2 ** 31 - 1, 1))
    for invariant in invariants:
        oPs._set(invariant, 0)
        mom_cons, on_shell, large_invs, small_invs = oPs.phasespace_consistency_check(invariants=invariants, silent=True)
        assert mom_cons
        assert on_shell
        assert large_invs == []
        assert small_invs == [invariant]
