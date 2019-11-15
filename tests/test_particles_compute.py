#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import mpmath
import sympy

from sympy.functions.special.tensor_functions import LeviCivita

from lips import Particles

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_particles_compute_lNB():

    oParticles = Particles(6)

    assert(abs(oParticles.compute("⟨1|(2-3)|4]") -
               (oParticles.compute("⟨1|2⟩") * oParticles.compute("[2|4]") - oParticles.compute("⟨1|3⟩") * oParticles.compute("[3|4]"))) < 10 ** -290)

    assert(abs(oParticles.compute("⟨1|(2+3)|4]") -
               (oParticles.compute("⟨1|2⟩") * oParticles.compute("[2|4]") + oParticles.compute("⟨1|3⟩") * oParticles.compute("[3|4]"))) < 10 ** -290)


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

    assert(abs(tr5 - oParticles.compute("tr5_1234")) < 10 ** -290)
