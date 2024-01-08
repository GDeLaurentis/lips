import numpy
import itertools
import re

from math import sqrt
from lips.algebraic_geometry.tools import lips_invariant_symbols, conversionIdeal
from syngular import Ideal, Ring


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class SpinorIdeal(Ideal):

    def __init__(self, ring_or_multiplicity, generators_or_invariants, momentum_conservation=None):
        """Initialises a fully analytical Ideal, either from a tuple of invariants, or from a list of generators."""

        if isinstance(ring_or_multiplicity, int):
            self.multiplicity = ring_or_multiplicity
        elif isinstance(ring_or_multiplicity, Ring):
            self.multiplicity = (1 + int(sqrt(4 * len(ring_or_multiplicity.variables) + 1))) // 2
        else:
            raise Exception("Invalid SpinorIdeal intialisation.")

        if type(generators_or_invariants) is tuple:
            from lips.algebraic_geometry.covariant_ideal import LipsIdeal
            oConversionIdeal = conversionIdeal(self.multiplicity)
            oComponentIdeal = LipsIdeal(self.multiplicity, generators_or_invariants, momentum_conservation)
            I = oConversionIdeal + oComponentIdeal
            J = I.eliminate(range(1, self.multiplicity * 4))
            generators = J.minbase

        elif type(generators_or_invariants) is list:
            generators = generators_or_invariants

        else:
            raise Exception("Invalid SpinorIdeal intialisation.")

        if isinstance(ring_or_multiplicity, int):
            super().__init__(Ring('0', lips_invariant_symbols(self.multiplicity), 'dp'), generators)
        elif isinstance(ring_or_multiplicity, Ring):
            super().__init__(ring_or_multiplicity, generators)

    def __call__(self, *args):
        if isinstance(args[0], str) and isinstance(args[1], bool):
            return self.image(args)
        else:
            raise NotImplementedError("LipsIdeal called with args: ", args)

    def image(self, rule):
        image_ideal = SpinorIdeal(self.ring, [invariant_poly_image(poly, rule) for poly in self.generators])
        if hasattr(self, "__name__"):
            image_ideal.__name__ = self.__name__ + str(rule)
        return image_ideal

    def to_momentum_and_schouten_qring(self):
        oZeroIdeal = SpinorIdeal(self.multiplicity, ())
        self.to_qring(oZeroIdeal)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def indices(multiplicity):
    return list(range(1, multiplicity + 1))


def indices_2d(multiplicity):
    pairs = list(itertools.combinations(indices(multiplicity), 2))
    return numpy.array([[pairs.index((i, j)) + 1 if (i, j) in pairs else -(pairs.index((j, i)) + 1) if (j, i) in pairs else 0
                         for j in indices(multiplicity)] for i in indices(multiplicity)])


def invariant_poly_image(polynomial, rule):
    permutation = tuple(entry - 1 for entry in map(int, tuple(rule[0])))
    multiplicity = len(rule[0])
    signed_permutation_2d = indices_2d(multiplicity)[permutation, :][:, permutation][numpy.triu_indices(multiplicity, k=1)]
    unsigned_permutation_2d = abs(signed_permutation_2d)
    polynomial = re.sub(r"([AB])(\d+)", lambda match: match.group(1) + str(unsigned_permutation_2d[int(match.group(2)) - 1])
                        if signed_permutation_2d[int(match.group(2)) - 1] > 0 else
                        "(-" + match.group(1) + str(unsigned_permutation_2d[int(match.group(2)) - 1]) + ")", polynomial)
    if rule[1] is True:
        polynomial = polynomial.replace("A", "C").replace("B", "A").replace("C", "B")
    return polynomial
