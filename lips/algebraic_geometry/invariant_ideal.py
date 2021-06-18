# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import sqrt
from lips.algebraic_geometry.tools import lips_invariant_symbols, conversionIdeal
from syngular import Ideal, Ring, QuotientRing


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

    def to_momentum_and_schouten_qring(self):
        oZeroIdeal = SpinorIdeal(self.multiplicity, ())
        qring = QuotientRing(Ring('0', lips_invariant_symbols(self.multiplicity), 'dp'), oZeroIdeal)
        self.ring = qring
