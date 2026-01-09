import numpy
import mpmath

from pycoretools import flatten
from lips import Particle

mpmath.mp.dps = 300


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_rank_two_spinor_setter():
    oParticle = Particle()
    assert all([abs(entry1 - entry2) < 10 ** -290 for entry1, entry2 in zip(flatten(oParticle.r2_sp_b), flatten(numpy.tensordot(oParticle.r_sp_d, oParticle.l_sp_d, axes=(1, 0))))])
