# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mpmath
import sympy
import numpy

from sympy.functions.special.tensor_functions import LeviCivita

from lips import Particles
from lips.tools import pSijk, pDijk


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_particles_compute_ldot():
    oParticles = Particles(5)
    assert(abs(oParticles.ldot(1, 2) - numpy.trace(numpy.dot(oParticles[1].r2_sp, oParticles[2].r2_sp_b)) / 2) < 10 ** -290)


def test_particles_compute_lNB():
    oParticles = Particles(6)
    assert(abs(oParticles("⟨1|2-3|4]") - oParticles("⟨1|2⟩[2|4]-⟨1|3⟩[3|4]")) < 10 ** -290)
    assert(abs(oParticles("⟨1|2+3|4]") - oParticles("⟨1|2⟩[2|4]+⟨1|3⟩[3|4]")) < 10 ** -290)


def test_particles_compute_Mandelstam():
    """Test computation of Mandelsta w.r.t. summed particles object."""
    oParticles = Particles(9)
    temp_string = "s_1234"
    ijk = list(map(int, pSijk.findall(temp_string)[0]))
    assert sum([oParticles[_i] for _i in ijk]).mass == oParticles(temp_string)


def test_particles_compute_three_mass_gram():
    """Test computation of 3-mass Gram w.r.t. summed particles object."""
    oParticles = Particles(9)
    temp_string = "Δ_135|249|678"
    match_list = pDijk.findall(temp_string)[0]
    if match_list[0] == '':
        NonOverlappingLists = [list(map(int, corner)) for corner in match_list[1:]]
    else:
        NonOverlappingLists = oParticles.ijk_to_3NonOverlappingLists(list(map(int, match_list[0])))
    temp_oParticles = oParticles.cluster(NonOverlappingLists)
    assert temp_oParticles.ldot(1, 2)**2 - temp_oParticles.ldot(1, 1) * temp_oParticles.ldot(2, 2) == oParticles(temp_string)


def test_particles_compute_tr5_1234():
    oParticles = Particles(5)
    tr5 = 0
    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                for l in range(0, 4):
                    tr5 += mpmath.mpc(sympy.simplify(4j *
                                                     LeviCivita(i, j, k, l) *
                                                     oParticles[1].four_mom[i] *
                                                     oParticles[2].four_mom[j] *
                                                     oParticles[3].four_mom[k] *
                                                     oParticles[4].four_mom[l]))
    assert(abs(tr5 - oParticles("tr5_1234")) < 10 ** -290)
