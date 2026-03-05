import re
import numpy
import sympy
import functools

from collections import defaultdict

from pycoretools import flatten
from lips.algebraic_geometry.tools import lips_covariant_symbols, lips_invariant_symbols, conversionIdeal
from lips.algebraic_geometry.invariant_ideal import SpinorIdeal
from syngular import Ideal, Ring


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class LipsIdeal(Ideal):
    """Lorentz Covariant Ideal - based on spinor components."""

    def __init__(self, ring_or_multiplicity, generators_or_covariants, momentum_conservation=None, verbose=False):
        """Initialises a fully analytical Ideal, generators_or_covariants can be either already parsed in term spinor components or not."""
        from lips import Particles

        if isinstance(ring_or_multiplicity, int):
            self.multiplicity = ring_or_multiplicity
            ring = Ring('0', lips_covariant_symbols(self.multiplicity), 'dp')
        elif isinstance(ring_or_multiplicity, Ring):
            self.multiplicity = len(ring_or_multiplicity.variables) // 2 // 2
            ring = ring_or_multiplicity
        else:
            raise Exception("Invalid LipsIdeal intialisation.")

        try:
            if (isinstance(generators_or_covariants, (list, tuple)) and len(generators_or_covariants) == 0 and
               (momentum_conservation is True or momentum_conservation is None)):
                raise Exception("Add momentum conservation.")
            if (isinstance(generators_or_covariants, (list, tuple)) and len(generators_or_covariants) > 0 and
               any([symbol in eq for symbol in ['[', ']', '|', '<', '>', '⟨', '⟩', ] for eq in generators_or_covariants])):
                raise Exception("Needs parsing.")
            # already in the ring variables
            super().__init__(ring, generators_or_covariants)
        except Exception:
            # parse first the spinor expressions
            oParticles = Particles(self.multiplicity)
            oParticles.make_analytical_d()
            generators = []
            for covariant in generators_or_covariants:
                # TODO: remove 4 * when https://github.com/sympy/sympy/pull/28139 is accepted
                poly_or_polys = 4 * oParticles(covariant)
                if hasattr(poly_or_polys, 'shape'):
                    polys = flatten(poly_or_polys)
                    for poly in polys:
                        generators += [str(sympy.Poly(sympy.expand(poly))).replace("Poly(", "").split(", ")[0]]
                else:
                    generators += [str(sympy.Poly(sympy.expand(poly_or_polys))).replace("Poly(", "").split(", ")[0]]
            if momentum_conservation is True or momentum_conservation is None:
                generators += [str(sympy.Poly(entry)).replace("Poly(", "").split(", ")[0] for entry in flatten(oParticles.total_mom)]
            super().__init__(ring, generators)

        if verbose:
            print("Initialized Lips Ideal:\n", repr(self))

    def __contains__(self, covariant):
        """Extends ideal membership to Lorentz covariant expressions computable with lips."""
        from lips import Particles
        oParticles = Particles(self.multiplicity)
        oParticles.make_analytical_d()
        try:
            poly_or_polys = 4 * oParticles(covariant)  # TODO: remove 4 * when https://github.com/sympy/sympy/pull/28139 is accepted
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

    @functools.cached_property
    def equivalenceClass(self):
        from ..symmetries import all_symmetries
        all_perms_self = {symmetry: self(*symmetry) for symmetry in all_symmetries(self.multiplicity)}
        value_to_keys = defaultdict(list)
        for k, v in all_perms_self.items():
            value_to_keys[v].append(k)
        distinct_perms_self = {keys[0]: val for val, keys in value_to_keys.items()}
        return distinct_perms_self

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
