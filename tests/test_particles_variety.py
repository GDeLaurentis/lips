# -*- coding: utf-8 -*-

import numpy
import pytest

from lips import Particles
from lips.fields.field import Field
from lips.algebraic_geometry.tools import singular_found


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.skipif(not singular_found, reason="singular not found")
def test_5_point_codim1_variety_padic():
    padic = Field('padic', 2 ** 31 - 1, 4)
    oParticles = Particles(5, field=padic)
    oParticles._singular_variety(("[1|2]", ), (1, ), verbose=True)
    assert abs(oParticles("[1|2]")).n == 1


@pytest.mark.skipif(not singular_found, reason="singular not found")
def test_5_point_codim1_variety_mpc():
    mpc = Field('mpc', 0, 300)
    oParticles = Particles(5, field=mpc)
    oParticles._singular_variety(("[1|2]", ), (10 ** -30, ), verbose=True)
    assert numpy.isclose(complex(oParticles("[1|2]")), 10 ** -30)


@pytest.mark.skipif(not singular_found, reason="singular not found")
def test_5_point_codim2_variety_padic():
    padic = Field('padic', 2 ** 31 - 1, 4)
    oParticles = Particles(5, field=padic)
    oParticles._singular_variety(("⟨2|1+5|2]", "⟨3|1+2|3]", ), (1, 2, ), verbose=True)
    assert abs(oParticles("⟨2|1+5|2]")).n == 1
    assert abs(oParticles("⟨3|1+2|3]")).n == 2


@pytest.mark.skipif(not singular_found, reason="singular not found")
def test_5_point_codim2_variety_mpc():
    mpc = Field('mpc', 0, 300)
    oParticles = Particles(5, field=mpc)
    oParticles._singular_variety(("⟨2|1+5|2]", "⟨3|1+2|3]", ), (10 ** -30, 2 * 10 ** -30, ), verbose=True)
    assert numpy.isclose(complex(oParticles("⟨2|1+5|2]")), 10 ** -30)
    assert numpy.isclose(complex(oParticles("⟨3|1+2|3]")), 2 * 10 ** -30)
