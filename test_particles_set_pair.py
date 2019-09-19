#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import pytest

from particles import Particles

from antares.core.invariants import Invariants
from antares.core.tools import mapThreads, retry, myException
from antares.core.settings import settings


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.parametrize(
    "multiplicity, type1, type2, expected_failures, expected_length",
    [
        (5, "invs_2", "invs_2", 0, 380),
        (5, "invs_2", "invs_N", 24, 320),
        (5, "invs_3", "invs_3", 76, 90),
        (6, "invs_2", "invs_2", 0, 870),
        (6, "invs_2", "invs_N", 48, 4950),
        (6, "invs_2", "invs_s", 0, 300),
        (6, "invs_2", "invs_D", 0, 60),
        (6, "invs_3", "invs_3", 204, 8556),
        (6, "invs_3", "invs_s", 0, 930),
        (6, "invs_3", "invs_D", 40, 186),
        (6, "invs_s", "invs_s", 0, 90),
        (6, "invs_s", "invs_D", 8, 20),
    ]
)
def test_particles_set(multiplicity, type1, type2, expected_failures, expected_length):

    oInvariants = Invariants(multiplicity, no_cached=True)
    tuples = [(inv1, inv2) for inv1 in eval("oInvariants." + type1) for inv2 in eval("oInvariants." + type2) if inv1 != inv2]

    TrueOrFalseList = mapThreads(DoubleScalingsTestingInner, multiplicity, oInvariants.full, tuples, UseParallelisation=settings.UseParallelisation, Cores=settings.Cores)
    failed_counter = sum(1 for entry in TrueOrFalseList if entry is False or entry is None)

    assert(failed_counter == expected_failures and len(tuples) == expected_length)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@retry((myException,), max_tries=2, silent=True)
def DoubleScalingsTestingInner(n, invariants, _tuple):
    oParticles = Particles(n)
    return oParticles.set_pair(_tuple[0], 10 ** -28, _tuple[1], 10 ** -28)
    # if invariants is not None:
    #     _, _, _, small_invs = oParticles.phasespace_consistency_check(invariants)
