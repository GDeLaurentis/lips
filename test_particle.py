import numpy

from particle import Particle

from antares.core.tools import MinkowskiMetric, Pauli_bar, flatten


def test_rank_two_spinor_setter():
    oParticle = Particle()
    temp_four_mom = oParticle.four_mom
    p_lowered_index = numpy.dot(MinkowskiMetric, temp_four_mom)
    # Contraction of 4 momentum with Van der Waerden symbol == outproduct of rank 1 spinors
    r2_s_b_1 = numpy.tensordot(p_lowered_index, Pauli_bar, axes=(0, 0))
    r2_s_b_2 = numpy.tensordot(oParticle.r_sp_d, oParticle.l_sp_d, axes=(1, 0))
    assert(max(map(abs, flatten(r2_s_b_1 - r2_s_b_2))) < 10 ** -300)
