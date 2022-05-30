# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import numpy
import sympy
import mpmath

from lips.fields import ModP, PAdic
from lips.algebraic_geometry.covariant_ideal import LipsIdeal
from lips.algebraic_geometry.tools import lex_groebner_solve, check_solutions, lips_covariant_symbols


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_SingularVariety:

    def _singular_variety(self, invariants, valuations=tuple(), generators=[], indepSetNbr=0, verbose=False):
        """Given invariants and valuations, generates a variety of dimension zero and solves for a ps point valuations away from the zero surface.
        If generators are given, they are used to construct the zero surface first, otherwise it is picked at random."""
        assert all([valuation > 0 for valuation in valuations])  # if valuations == tuple() return zero surface

        if self.field.name == "padic":
            prime, iterations = self.field.characteristic, self.field.digits
            padic_to_finite_field(self)  # work with p ** k finite field itaratively, since Singular can only handle % p
            if valuations == tuple():
                valuations = tuple(self.field.digits for inv in invariants)
        elif self.field.name == "finite field":
            prime, iterations = self.field.characteristic, 1
        elif self.field.name == "mpc":
            prime, iterations = None, 1 if valuations == tuple() else 2

        if generators == []:
            oIdeal = LipsIdeal(len(self), invariants)
        else:
            oIdeal = LipsIdeal(len(self), generators)
        # oIdeal.ring.ordering = 'lp'  # no need to set lex ordering here - it just tanks the performance

        # print(oIdeal.generators)

        indepSets = oIdeal.indepSets
        if verbose:
            print("Codimensions:", set(indepSet.count(0) - 4 for indepSet in indepSets))
            print("Number of indepSets:", len(indepSets))
        indepSet = indepSets[indepSetNbr]
        indepSymbols = [symbol for i, symbol in enumerate(lips_covariant_symbols(len(self))) if indepSet[i] == 1]
        if verbose:
            print("Chosen indepSet:", indepSet)

        self.make_analytical_d(indepVars=indepSet)
        oSemiNumericalIdeal = oIdeal.zero_dimensional_slice(self, invariants, valuations, prime=prime, iteration=0)

        # print(repr(oSemiNumericalIdeal))

        for iteration in range(iterations):

            # make sure ordering is lexicographical and there is no cached groebner_basis
            oSemiNumericalIdeal.ring.ordering = 'lp'
            if hasattr(oSemiNumericalIdeal, "groebner_basis"):
                del oSemiNumericalIdeal.groebner_basis
            # print(repr(oSemiNumericalIdeal), oSemiNumericalIdeal.primary_decomposition, oSemiNumericalIdeal.groebner_basis, len(oSemiNumericalIdeal.groebner_basis))
            root_dicts = lex_groebner_solve(oSemiNumericalIdeal.groebner_basis, prime=prime)
            check_solutions(oSemiNumericalIdeal.groebner_basis, root_dicts, prime=prime)

            root_dict = root_dicts[0]
            root_dict = {key: root_dict[key] for key in root_dict.keys() if key not in indepSymbols}

            if iteration < iterations - 1:
                for key in root_dict.keys():
                    root_dict[key] = root_dict[key] + (prime if prime is not None else 1) * key

            # print(root_dict)

            update_particles(self, root_dict)

            if iteration < iterations - 1:
                if prime is not None:
                    valuations = [valuation - 1 for valuation in valuations]
                invariants_valuations = list(filter(lambda x: x[1] > 0, zip(invariants, valuations)))
                if invariants_valuations != []:
                    invariants, valuations = zip(*invariants_valuations)
                else:
                    invariants, valuations = (), ()
                if verbose:
                    print("Invariants, valuations:", invariants, valuations)

                oSemiNumericalIdeal = oIdeal.zero_dimensional_slice(self, invariants, valuations, prime=prime, iteration=iteration + 1)

                # print(oSemiNumericalIdeal.generators)

                if len(oSemiNumericalIdeal.indepSets) == 0:
                    raise Exception("No independent set exists: is this the unit ideal?!")
                currentIndepSet = oSemiNumericalIdeal.indepSets[0]
                if verbose:
                    print("Chosen indepSet:", currentIndepSet)
                if currentIndepSet != indepSet:  # this happens only with padics
                    newIndepSymbols = tuple(symbol for i, symbol in enumerate(lips_covariant_symbols(len(self))) if currentIndepSet[i] == 1 and indepSet[i] == 0)
                    rand_dict = {newIndepSymbol: random.randrange(1, self.field.characteristic ** (self.field.digits - iteration)) for newIndepSymbol in newIndepSymbols}
                    update_particles(self, rand_dict)
                    indepSet = currentIndepSet
                    oSemiNumericalIdeal = oIdeal.zero_dimensional_slice(self, invariants, valuations, prime=prime, iteration=iteration + 1)
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
        if hasattr(sol[key], "free_symbols") and sol[key].free_symbols == set() and oParticles.field.name == "mpc":
            sol[key] = mpmath.mpc(sol[key])
    for i in range(1, len(oParticles) + 1):
        oParticles[i]._l_sp_d = numpy.array([[sol[sympy.symbols('c%i' % i)], sol[sympy.symbols('d%i' % i)]]], dtype=object)
        oParticles[i]._l_sp_d_to_l_sp_u()
        oParticles[i].r_sp_d = numpy.array([[sol[sympy.symbols('a%i' % i)]], [sol[sympy.symbols('b%i' % i)]]], dtype=object)
