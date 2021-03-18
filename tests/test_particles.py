# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from lips.fields import Field
from lips import Particles


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


mpc = Field('mpc', 0, 300)
modp = Field('finite field', 2 ** 31 - 1, 1)


@pytest.mark.parametrize(
    "multiplicity, field",
    [
        (4, mpc), (4, modp),
        (5, mpc), (5, modp),
        (6, mpc), (6, modp),
        (7, mpc), (7, modp),
    ]
)
def test_particles_instantiation(multiplicity, field):
    oPs = Particles(multiplicity, field=field)
    assert(oPs.momentum_conservation_check())
    assert(oPs.onshell_relation_check())


@pytest.mark.parametrize(
    "multiplicity, real_momenta, axis",
    [
        (5, False, 1),
        (5, False, 2),
        (5, True, 1),
        (5, True, 2),
        (5, True, 3),
        (6, False, 1),
        (6, False, 2),
        (6, True, 1),
        (6, True, 2),
        (6, True, 3),
    ]
)
def test_particles_fix_mom_cons(multiplicity, real_momenta, axis):

    oParticles = Particles(multiplicity, real_momenta=real_momenta, fix_mom_cons=False)
    oParticles.fix_mom_cons(real_momenta=real_momenta, axis=axis)

    assert(oParticles.momentum_conservation_check())
    assert(oParticles.onshell_relation_check())
