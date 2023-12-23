import sympy
import itertools

from syngular import Ideal, Ring


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def lips_covariant_symbols(multiplicity):
    """Returns a list of sympy symbols: [a1, b1, c1, d1, ...].
       If using make_analytical_d we have a1=oPs[1].r_sp_d[0, 0], b1=oPs[1].r_sp_d[1, 0], c1=oPs[1].l_sp_d[0, 0], d1=oPs[1].l_sp_d[0, 1]."""
    la = sympy.symbols('a1:{}'.format(multiplicity + 1))
    lb = sympy.symbols('b1:{}'.format(multiplicity + 1))
    lc = sympy.symbols('c1:{}'.format(multiplicity + 1))
    ld = sympy.symbols('d1:{}'.format(multiplicity + 1))
    iters = map(iter, [la, lb, lc, ld])
    return tuple(next(it) for it in itertools.islice(itertools.cycle(iters), 4 * multiplicity))


def lips_invariant_symbols(multiplicity):
    """Returns a list of sympy symbols: [A1, A2, ..., B1, B2, ...].
       With A1 = ⟨1|2⟩, A2 = ⟨1|3⟩, ..., B1 = [1|2], B2 = [1|3], ..."""
    lA = sympy.symbols('A1:{}'.format(multiplicity * (multiplicity - 1) // 2 + 1))
    lB = sympy.symbols('B1:{}'.format(multiplicity * (multiplicity - 1) // 2 + 1))
    return lA + lB


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def conversionIdeal(multiplicity):
    ring = Ring('0', lips_covariant_symbols(multiplicity) + lips_invariant_symbols(multiplicity), 'dp')
    from lips import Particles
    oParticles = Particles(multiplicity)
    oParticles.make_analytical_d()
    indices = range(1, multiplicity + 1)
    pairs = list(itertools.combinations(indices, 2))
    spas = ["⟨{}|{}⟩".format(*pair) for pair in pairs]
    spbs = ["[{}|{}]".format(*pair) for pair in pairs]
    generators = [str(oParticles(spa)) + "-A{}".format(i + 1) for i, spa in enumerate(spas)] + [str(oParticles(spb)) + "-B{}".format(i + 1) for i, spb in enumerate(spbs)]
    return Ideal(ring, generators)
