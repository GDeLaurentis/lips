import re
import numpy
import sympy

from lips.tools import flatten
from lips.algebraic_geometry.tools import lips_covariant_symbols, lips_invariant_symbols, conversionIdeal
from lips.algebraic_geometry.invariant_ideal import SpinorIdeal
from syngular import Ideal, Ring


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class LipsIdeal(Ideal):
    """Lorentz Covariant Ideal - based on spinor components."""

    def __init__(self, ring_or_multiplicity, generators_or_covariants, momentum_conservation=None):
        """Initialises a fully analytical Ideal, either from a tuple of covariants, or from a list of generators."""

        if isinstance(ring_or_multiplicity, int):
            self.multiplicity = ring_or_multiplicity
        elif isinstance(ring_or_multiplicity, Ring):
            self.multiplicity = len(ring_or_multiplicity.variables) // 2 // 2
        else:
            raise Exception("Invalid LipsIdeal intialisation.")

        if type(generators_or_covariants) is tuple:
            from lips import Particles
            oParticles = Particles(self.multiplicity)
            oParticles.make_analytical_d()
            generators = []
            for covariant in generators_or_covariants:
                poly_or_polys = 4 * oParticles(covariant)
                if hasattr(poly_or_polys, 'shape'):
                    polys = flatten(poly_or_polys)
                    for poly in polys:
                        generators += [str(sympy.Poly(sympy.expand(poly))).replace("Poly(", "").split(", ")[0]]
                else:
                    generators += [str(sympy.Poly(sympy.expand(poly_or_polys))).replace("Poly(", "").split(", ")[0]]
            if momentum_conservation is True or momentum_conservation is None:
                generators += [str(sympy.Poly(entry)).replace("Poly(", "").split(", ")[0] for entry in flatten(oParticles.total_mom)]

        elif type(generators_or_covariants) is list:
            generators = generators_or_covariants

        else:
            raise Exception("Invalid LipsIdeal intialisation.")

        if isinstance(ring_or_multiplicity, int):
            super().__init__(Ring('0', lips_covariant_symbols(self.multiplicity), 'dp'), generators)
        elif isinstance(ring_or_multiplicity, Ring):
            super().__init__(ring_or_multiplicity, generators)

    def __contains__(self, covariant):
        """Extends ideal membership to Lorentz covariant expressions computable with lips."""
        from lips import Particles
        oParticles = Particles(self.multiplicity)
        oParticles.make_analytical_d()
        try:
            poly_or_polys = 4 * oParticles(covariant)
        except TypeError:
            poly_or_polys = covariant
        if isinstance(poly_or_polys, numpy.ndarray):
            return all(super(LipsIdeal, self).__contains__(poly) for poly in flatten(poly_or_polys))
        else:
            return super().__contains__(poly_or_polys)

    def __call__(self, *args):
        if isinstance(args[0], str) and isinstance(args[1], bool):
            return self.image(args)
        else:
            raise NotImplementedError("LipsIdeal called with args: ", args)

    def image(self, rule):
        newIdeal = LipsIdeal(self.ring, [covariant_poly_image(poly, rule) for poly in self.generators])
        newIdeal.rule = rule
        return newIdeal

    def to_mom_cons_qring(self):
        oZeroIdeal = LipsIdeal(self.multiplicity, ())
        self.to_qring(oZeroIdeal)

    def invariant_slice(self):
        oConversionIdeal = conversionIdeal(self.multiplicity)
        I = oConversionIdeal + self
        J = I.eliminate(range(0, self.multiplicity * 4))
        return SpinorIdeal(Ring('0', lips_invariant_symbols(self.multiplicity), 'dp'), J.generators)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def covariant_poly_image(polynomial, rule):
    polynomial = re.sub(r"(?<=[abcd])(\d)", lambda match: rule[0][int(match.group(0)) - 1], polynomial)
    if rule[1] is True:
        polynomial = polynomial.replace("a", "A").replace("b", "B").replace("c", "a").replace("d", "b").replace("A", "c").replace("B", "d")
    return polynomial
