#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from lips import Particles


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_particles_compute_lNB():

    oParticles = Particles(6)
    oParticles.fix_mom_cons()

    assert(abs(oParticles.compute("⟨1|(2-3)|4]") -
               (oParticles.compute("⟨1|2⟩") * oParticles.compute("[2|4]") - oParticles.compute("⟨1|3⟩") * oParticles.compute("[3|4]"))) < 10 ** -290)

    assert(abs(oParticles.compute("⟨1|(2+3)|4]") -
               (oParticles.compute("⟨1|2⟩") * oParticles.compute("[2|4]") + oParticles.compute("⟨1|3⟩") * oParticles.compute("[3|4]"))) < 10 ** -290)
