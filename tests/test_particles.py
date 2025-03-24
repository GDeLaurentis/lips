# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import pytest
import pickle

from lips.fields import Field
from lips import Particles


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


mpc = Field('mpc', 0, 300)
modp = Field('finite field', 2 ** 31 - 1, 1)
padic = Field('padic', 2 ** 31 - 1, 6)


@pytest.mark.parametrize(
    "multiplicity, field",
    [
        (4, mpc), (4, modp), (4, padic),
        (5, mpc), (5, modp), (5, padic),
        (6, mpc), (6, modp), (6, padic),
        (7, mpc), (7, modp), (7, padic),
    ]
)
def test_particles_instantiation(multiplicity, field):
    oPs = Particles(multiplicity, field=field)
    assert oPs.momentum_conservation_check()
    assert oPs.onshell_relation_check()


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

    assert oParticles.momentum_conservation_check()
    assert oParticles.onshell_relation_check()


def test_spinor_item_setter():
    oPs = Particles(8)
    a, b, c, d = numpy.array([[1, 2]]), numpy.array([[2], [-1]]), numpy.array([[3, 4]]), numpy.array([[-4], [3]])
    oPs["[5|"] = a
    assert numpy.all(oPs["|5]"] == b)
    oPs["|5⟩"] = d
    assert numpy.all(oPs["⟨5|"] == c)


def test_equality_after_pickle():
    # Note that pickle uses __reduce__ to dump bytecode info of the object.
    oPs = Particles(6, field=Field("finite field", 2 ** 31 - 1, 1), seed=0)
    oPs2 = oPs.image(('132456', False))
    oPs3 = oPs.image(('132456', False))
    assert oPs2 == oPs3
    assert pickle.dumps(oPs2) == pickle.dumps(oPs3)


def test_particles_cluster_with_massive_fermions():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=0)
    oPs._singular_variety(("⟨34⟩+[34]", "⟨34⟩-⟨56⟩", "⟨56⟩+[56]"), (1, 1, 1))
    oPs.mt2 = oPs("s_34")
    oPs.mt = - oPs("⟨34⟩")
    oPsClusteredTensor = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', all), (4, 'd', all)))
    oPsClusteredTensor11 = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 1), (4, 'd', 1)))
    oPsClusteredTensor12 = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 1), (4, 'd', 2)))
    oPsClusteredTensor21 = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 2), (4, 'd', 1)))
    oPsClusteredTensor22 = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 2), (4, 'd', 2)))
    assert (oPsClusteredTensor("⟨3|5|4]") == numpy.array([[oPsClusteredTensor11("⟨3|5|4]"), oPsClusteredTensor12("⟨3|5|4]")],
                                                          [oPsClusteredTensor21("⟨3|5|4]"), oPsClusteredTensor22("⟨3|5|4]")]])).all()


def test_Dirac_equation():
    oPs = Particles(6, field=Field("finite field", 2 ** 31 - 1, 1), seed=None)
    oPsClustered = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], ], massive_fermions=((3, 'u', 1), (4, 'd', 1)))
    oPsClustered.m = - oPs("⟨3|4⟩")
    oPsClustered.mb = oPs("[3|4]")
    assert oPsClustered("s_3") == oPsClustered.m * oPsClustered.mb
    assert (oPsClustered("|3|3⟩") == - oPsClustered("m|3]")).all()
    assert (oPsClustered("|3|3]") == - oPsClustered("mb|3⟩")).all()
    assert (oPsClustered("⟨3|3|") == oPsClustered("m[3|")).all()
    assert (oPsClustered("[3|3|") == oPsClustered("mb⟨3|")).all()
