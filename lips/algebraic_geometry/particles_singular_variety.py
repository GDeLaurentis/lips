import numpy

from pyadic import ModP, PAdic

from ..tools import flatten

from .covariant_ideal import LipsIdeal


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_SingularVariety:

    def _singular_variety(self, invariants, valuations=(), generators=[], indepSetNbr=None, seed=None, verbose=False, no_base_point=False):

        from ..particles import Particles

        if self.field.name != "finite field":
            oPsAnalytical = Particles(self.multiplicity)
            oPsAnalytical.make_analytical_d()
            directions = flatten([oPsAnalytical(invariant) for invariant in invariants])
        else:
            directions = None

        if generators == []:
            generators = invariants

        oLipsIdeal = LipsIdeal(self.multiplicity, generators)
        oLipsIdeal.to_mom_cons_qring()

        point_dict = oLipsIdeal.point_on_variety(
            self.field, base_point={} if no_base_point else self.analytical_subs_d(), directions=directions,
            valuations=valuations, indepSetNbr=indepSetNbr, seed=seed, verbose=verbose)

        update_particles(self, point_dict)

    @classmethod
    def from_singular_variety(cls, multiplicity, field, invariants, valuations=(), generators=[], indepSetNbr=None, seed=None, verbose=False):
        oPs = cls(multiplicity, field=field)  # dummy point
        oPs._singular_variety(invariants, valuations=valuations, generators=generators, indepSetNbr=indepSetNbr, seed=seed, verbose=verbose, no_base_point=True)
        return oPs

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def padic_to_finite_field(oParticles):
    # prime, digits = oParticles.field.characteristic, oParticles.field.digits
    for oParticle in oParticles:
        oParticle._l_sp_d = numpy.array([[ModP(oParticle.l_sp_d[0, 0]), ModP(oParticle.l_sp_d[0, 1])]], dtype=object)
        oParticle._l_sp_d_to_l_sp_u()
        oParticle.r_sp_d = numpy.array([[ModP(oParticle.r_sp_d[0, 0])], [ModP(oParticle.r_sp_d[1, 0])]], dtype=object)
    # oParticles.field = Field('finite field', prime, 0)


def finite_field_to_padic(oParticles):
    for oParticle in oParticles:   # this needs from_addition = True otherwise it wrongly extends precision on instantiation of soft components
        oParticle._l_sp_d = numpy.array([[PAdic(oParticle.l_sp_d[0, 0], oParticles.field.characteristic, oParticles.field.digits, from_addition=True),
                                          PAdic(oParticle.l_sp_d[0, 1], oParticles.field.characteristic, oParticles.field.digits, from_addition=True)]])
        oParticle._l_sp_d_to_l_sp_u()
        oParticle.r_sp_d = numpy.array([[PAdic(oParticle.r_sp_d[0, 0], oParticles.field.characteristic, oParticles.field.digits, from_addition=True)],
                                        [PAdic(oParticle.r_sp_d[1, 0], oParticles.field.characteristic, oParticles.field.digits, from_addition=True)]])


def update_particles(oParticles, dictionary):
    for i in range(1, len(oParticles) + 1):
        oParticles[i]._l_sp_d = numpy.array([[dictionary[f'c{i}'], dictionary[f'd{i}']]], dtype=object)
        oParticles[i]._l_sp_d_to_l_sp_u()
        oParticles[i].r_sp_d = numpy.array([[dictionary[f'a{i}']], [dictionary[f'b{i}']]], dtype=object)
