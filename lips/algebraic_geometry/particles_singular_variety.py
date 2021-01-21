# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import numpy
import sympy

from lips.fields import ModP, PAdic
from lips.algebraic_geometry.ideal import LipsIdeal
from lips.algebraic_geometry.tools import lex_groebner_solve, check_solutions, lips_symbols


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_SingularVariety:

    def _singular_variety(self, invariants, valuations, verbose=False):
        """Given generators and valuations, generates a variety of dimension zero and solves for a ps point valuations away from the zero surface."""
        assert all([valuation > 0 for valuation in valuations])

        if self.field.name == "padic":
            prime, iterations = self.field.characteristic, self.field.digits
            padic_to_finite_field(self)  # work with p ** k finite field itaratively, since Singular can only handle % p
        else:
            prime, iterations = None, 1
            invariants = [invariant + "-{}".format(valuation) for (invariant, valuation) in zip(invariants, valuations)]

        oAnalyticalIdeal = LipsIdeal(self, invariants)
        indepSets = oAnalyticalIdeal.indepSets
        if verbose:
            print("Codimension:", set(indepSet.count(0) - 4 for indepSet in indepSets))
        indepSet = indepSets[0]
        if verbose:
            print("Chosen indepSet:", indepSet)
        oSemiNumericalIdeal = LipsIdeal(self, invariants, indepVars=indepSet, prime=prime)

        for iteration in range(iterations):

            root_dicts = lex_groebner_solve(oSemiNumericalIdeal.groebner_basis, prime=prime)
            check_solutions(oSemiNumericalIdeal.groebner_basis, root_dicts, prime=prime)

            root_dict = root_dicts[0]

            if iteration < iterations - 1:
                for key in root_dict.keys():
                    root_dict[key] = root_dict[key] + prime * key

            update_particles(self, root_dict)

            if iteration < iterations - 1:
                current_valuations = [valuation - (iteration + 1) for valuation in valuations]
                current_invariants = [invariant for i, invariant in enumerate(invariants) if current_valuations[i] > 0]

                oSemiNumericalIdeal.update_generators(current_invariants, iteration + 1, prime)

                currentIndepSet = oSemiNumericalIdeal.indepSets[0]
                if currentIndepSet != indepSet:
                    newIndepSymbols = tuple(symbol for i, symbol in enumerate(lips_symbols(len(self))) if currentIndepSet[i] == 1 and indepSet[i] == 0)
                    rand_dict = {newIndepSymbol: random.randrange(1, self.field.characteristic ** (self.field.digits - iteration)) for newIndepSymbol in newIndepSymbols}
                    update_particles(self, rand_dict)
                    indepSet = currentIndepSet
                    oSemiNumericalIdeal.update_generators(current_invariants, iteration + 1, prime)
        else:
            if self.field.name == "padic":
                finite_field_to_padic(self)


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
    sol = oParticles.analytical_subs_d()
    for key in sol:
        if key in dictionary and sol[key] == key:
            sol[key] = dictionary[key]
        elif key in dictionary:
            sol[key] = sol[key].subs(dictionary)
    for i in range(1, len(oParticles) + 1):
        oParticles[i]._l_sp_d = numpy.array([[sol[sympy.symbols('c%i' % i)], sol[sympy.symbols('d%i' % i)]]], dtype=object)
        oParticles[i]._l_sp_d_to_l_sp_u()
        oParticles[i].r_sp_d = numpy.array([[sol[sympy.symbols('a%i' % i)]], [sol[sympy.symbols('b%i' % i)]]], dtype=object)
