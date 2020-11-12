# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sympy
import subprocess

from copy import deepcopy
from lips.tools import flatten
from lips.algebraic_geometry.tools import lips_symbols, singular_clean_up_lines


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class LipsIdeal(object):

    def __init__(self, oParticles, invariants, momentum_conservation=True, indepVars=None, prime=None):
        oParticles = deepcopy(oParticles) if indepVars is None else oParticles
        oParticles.make_analytical_d(indepVars=indepVars)

        generators = [str(sympy.Poly(sympy.expand(4 * oParticles.compute(invariant)), modulus=prime)).replace("Poly(", "").split(", ")[0] for invariant in invariants]
        if momentum_conservation is True:
            generators += [str(sympy.Poly(entry, modulus=prime)).replace("Poly(", "").split(", ")[0] for entry in flatten(oParticles.total_mom)]

        self.oParticles = oParticles
        self.invariants = invariants
        self.generators = generators
        self.momentum_conservation = momentum_conservation

    @property
    def indepSets(self):
        singular_commands = ["ring r = " + self.oParticles.field.singular_notation + ", (" + ", ".join(map(str, lips_symbols(len(self.oParticles)))) + "), dp;",
                             "ideal i = " + ",".join(map(str, self.generators)) + ";",
                             "ideal gb = groebner(i);",
                             "print(indepSet(gb, 1));",
                             "$"]
        singular_command = "\n".join(singular_commands)
        # print(singular_command)
        test = subprocess.Popen(["timeout", "2", "Singular", "--quiet", "--execute", singular_command], stdout=subprocess.PIPE)
        output = test.communicate()[0]
        indepSets = [tuple(map(int, line.replace(" ", "").split(","))) for line in output.decode("utf-8").split("\n") if line not in singular_clean_up_lines and ":" not in line]
        return indepSets

    @property
    def groebner_basis(self):
        singular_commands = ["ring r = " + self.oParticles.field.singular_notation + ", (" + ", ".join(map(str, lips_symbols(len(self.oParticles)))) + "), lp;",
                             "ideal i = " + ",".join(map(str, self.generators)) + ";",
                             "ideal gb = groebner(i);",
                             "print(gb);",
                             "$"]
        singular_command = "\n".join(singular_commands)
        # print(singular_command)
        test = subprocess.Popen(["timeout", "2", "Singular", "--quiet", "--execute", singular_command], stdout=subprocess.PIPE)
        output = test.communicate()[0]
        output = [line.replace(",", "") for line in output.decode("utf-8").split("\n") if line not in singular_clean_up_lines]
        return output

    def update_generators(self, invariants, iteration, prime):
        generators = [str(sympy.Poly(sympy.expand(4 * self.oParticles.compute(invariant)) / prime ** iteration, modulus=prime)).replace("Poly(", "").split(", ")[0]
                      for invariant in invariants]
        if self.momentum_conservation is True:
            generators += [str(sympy.Poly(entry / prime ** iteration, modulus=prime)).replace("Poly(", "").split(", ")[0] for entry in flatten(self.oParticles.total_mom)]
        self.generators = generators
