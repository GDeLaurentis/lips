import numpy
from core.tools import MinkowskiMetric, Pauli_bar
from particle import Particle


def test_rank_two_spinor_setter():
    oParticle = Particle()
    temp_four_mom = oParticle.four_mom
    p_lowered_index = numpy.dot(MinkowskiMetric, temp_four_mom)
    r2_s_b_1 = numpy.tensordot(p_lowered_index, Pauli_bar, axes=(0, 0))
    r2_s_b_2 = numpy.tensordot(oParticle.r_sp_d, oParticle.l_sp_d, axes=(1, 0))
    assert(abs(numpy.max(r2_s_b_1 - r2_s_b_2)) < 10 ** -300)
