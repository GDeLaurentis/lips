# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mpmath
import sympy
import numpy
import pytest

from sympy.functions.special.tensor_functions import LeviCivita

from lips import Particles
from lips.tools import pSijk, pDijk
from lips.fields.field import Field

mpc = Field('mpc', 0, 300)
modp = Field('finite field', 2 ** 31 - 1, 1)
padic = Field('padic', 2 ** 31 - 1, 6)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_compute_ldot(field):
    oParticles = Particles(5, field=field)
    assert abs(oParticles.ldot(1, 2) - numpy.trace(numpy.dot(oParticles[1].r2_sp, oParticles[2].r2_sp_b)) / 2) <= field.tollerance


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_compute_lNB(field):
    oParticles = Particles(6, field=field)
    assert abs(oParticles("⟨1|2-3|4]") - oParticles("⟨1|2⟩[2|4]-⟨1|3⟩[3|4]")) <= field.tollerance
    assert abs(oParticles("⟨1|2+3|4]") - oParticles("⟨1|2⟩[2|4]+⟨1|3⟩[3|4]")) <= field.tollerance


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_compute_lNB_open(field):
    oParticles = Particles(7, field=field)
    assert numpy.all(abs(oParticles("|4+5|6+7|1]-|6+7|4+5|1]") - oParticles("|4]⟨4|6+7|1]+|5]⟨5|6+7|1]-|6]⟨6|4+5|1]-|7]⟨7|4+5|1]")) <= field.tollerance)
    assert numpy.all(abs(oParticles("[1|4+5|6+7|-[1|6+7|4+5|") - oParticles("[1|4+5|6⟩[6|+[1|4+5|7⟩[7|-[1|6+7|4⟩[4|-[1|6+7|5⟩[5|")) <= field.tollerance)


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
    assert abs(tr5 - oParticles("tr5_1234")) < 10 ** -290
