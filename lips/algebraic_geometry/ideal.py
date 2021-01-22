# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import sympy
import subprocess

from copy import deepcopy
from lips.tools import flatten
from lips.algebraic_geometry.tools import lips_symbols, singular_clean_up_lines


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class LipsIdeal(object):
    """Lorentz Invariant Ideal - based on co-variant spinor compontens."""

    def __init__(self, multiplicity, invariants_or_polynomials, momentum_conservation=True):
        """Initialises a fully analytical exact Ideal, either from a tuple of invariants, or from a list of polynomials."""
        self.multiplicity = multiplicity
        self.momentum_conservation = momentum_conservation

        if type(invariants_or_polynomials) is tuple:
            from lips import Particles
            oParticles = Particles(multiplicity)
            oParticles.make_analytical_d()
            self.generators = [str(sympy.Poly(sympy.expand(4 * oParticles.compute(invariant)))).replace("Poly(", "").split(", ")[0] for invariant in invariants_or_polynomials]
            if momentum_conservation is True:
                self.generators += [str(sympy.Poly(entry)).replace("Poly(", "").split(", ")[0] for entry in flatten(oParticles.total_mom)]

        elif type(invariants_or_polynomials) is list:
            self.generators = invariants_or_polynomials

        else:
            raise Exception("Invalid LipsIdeal intialisation.")

    def zero_dimensional_slice(self, oParticles, invariants, valuations, prime=None, iteration=0):
        """Returns a new ideal corresponding to a zero-dimensional (potentially perturbed) slice of the origial ideal self."""

        # regenerate the ideal (potentiallty losing branch information), beacuse: (floats) need to append perturbations to equations, (padics) may need to solve less equations.
        if iteration > 0:
            oSemiNumericalIdeal = LipsIdeal(len(oParticles), invariants)
            if prime is None:
                for i, valuation in enumerate(valuations):
                    oSemiNumericalIdeal.generators[i] = oSemiNumericalIdeal.generators[i] + str(-4 * sympy.sympify(valuation))
        else:
            oSemiNumericalIdeal = deepcopy(self)

        subs = oParticles.analytical_subs_d()
        # print("Subs:", subs)
        oSemiNumericalIdeal.generators = sympy.sympify(oSemiNumericalIdeal.generators)
        oSemiNumericalIdeal.generators = [sympy.expand(generator.subs(subs)) for generator in oSemiNumericalIdeal.generators]
        oSemiNumericalIdeal.generators = list(filter(lambda x: x != 0, oSemiNumericalIdeal.generators))

        if prime is None:
            oSemiNumericalIdeal.generators = [str(generator) for generator in oSemiNumericalIdeal.generators]
        else:
            oSemiNumericalIdeal.generators = [re.sub(r"(?<![a-z])(\d+)", lambda match: str(int(match.group(1)) // prime ** iteration % prime), str(generator))
                                              for generator in oSemiNumericalIdeal.generators]
            oSemiNumericalIdeal.generators = sympy.sympify(oSemiNumericalIdeal.generators)
            oSemiNumericalIdeal.generators = list(filter(lambda x: x != 0, oSemiNumericalIdeal.generators))

        oSemiNumericalIdeal.oParticles = oParticles
        return oSemiNumericalIdeal

    @property
    def singular_field_notation(self):
        if hasattr(self, "oParticles"):
            return self.oParticles.field.singular_notation
        else:
            return "0"

    @property
    def indepSets(self):
        singular_commands = ["ring r = " + self.singular_field_notation + ", (" + ", ".join(map(str, lips_symbols(self.multiplicity))) + "), dp;",
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
        singular_commands = ["ring r = " + self.singular_field_notation + ", (" + ", ".join(map(str, lips_symbols(self.multiplicity))) + "), lp;",
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

    @property
    def primary_decomposition(self):
        singular_commands = ["LIB \"primdec.lib\";",
                             "ring r = 0, (" + ", ".join(map(str, lips_symbols(self.multiplicity))) + "), dp;",
                             "ideal i = " + ",".join(map(str, self.generators)) + ";",
                             "ideal gb = groebner(i);",
                             "def pr = primdecGTZ(gb);",
                             "print(pr);"]
        singular_command = "\n".join(singular_commands)
        # print(singular_command)
        test = subprocess.Popen(["timeout", "600", "Singular", "--quiet", "--execute", singular_command], stdout=subprocess.PIPE)
        output = test.communicate()[0]
        output = [line.replace(",", "") for line in output.decode("utf-8").split("\n") if line not in singular_clean_up_lines]
        # print(output)  # ['halt 1']

        def clean_up(string):
            string = re.sub(r"_\[\d+\]=", "", string)
            string = re.sub(r"\[\d+\]:", "|", string)
            return string.replace(" ", "")

        primary_decomposed = ",".join(list(map(clean_up, output))).replace("|,", "|").replace(",|", "|").replace("|,", "|").split("||")
        primary_decomposed = list(filter(lambda x: x != '', primary_decomposed))
        primary_decomposed = [entry.split("|") for entry in primary_decomposed]
        primary_decomposed = [(entry[0].split(","), entry[1].split(",")) for entry in primary_decomposed]
        return primary_decomposed

    def __contains__(self, invariant):
        """Implements ideal membership."""
        from lips import Particles
        oParticles = Particles(self.multiplicity)
        oParticles.make_analytical_d()
        invariant = str(sympy.expand(4 * oParticles.compute(invariant)))
        singular_commands = ["ring r = 0, (" + ", ".join(map(str, lips_symbols(len(oParticles)))) + "), dp;",
                             "ideal i = " + ",".join(map(str, self.generators)) + ";",
                             "ideal gb = groebner(i);",
                             "poly f = " + invariant + ";",
                             "print(reduce(f,gb));"
                             "$"]
        singular_command = "\n".join(singular_commands)
        # print(singular_command)
        test = subprocess.Popen(["timeout", "2", "Singular", "--quiet", "--execute", singular_command], stdout=subprocess.PIPE)
        output = test.communicate()[0]
        output = [line.replace(",", "") for line in output.decode("utf-8").split("\n") if line not in singular_clean_up_lines]
        # print(output)
        if output == ['0']:
            return True
        else:
            return False
