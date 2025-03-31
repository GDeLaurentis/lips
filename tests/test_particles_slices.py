import sympy

from lips import Particles
from syngular import Field


def test_slice_covariant_algo():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=None)
    oPs.univariate_slice(extra_constraints=("⟨34⟩+[34]", "⟨34⟩-⟨56⟩", "⟨56⟩+[56]"), seed=0, indepSets=None, algorithm='covariant', verbose=False)

    assert isinstance(oPs("[12]"), sympy.Expr)
    assert sympy.expand(oPs("[12]"), modulus=oPs.field.characteristic).as_poly().degree() == 1


def test_slice_generic_algo():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=None)
    oPs.univariate_slice(extra_constraints=("⟨34⟩+[34]", "⟨34⟩-⟨56⟩", "⟨56⟩+[56]"), seed=0, indepSets=None, algorithm='generic', verbose=False)

    assert isinstance(oPs("[12]"), sympy.Expr)
    assert sympy.expand(oPs("[12]"), modulus=oPs.field.characteristic).as_poly().degree() == 2


def test_slice_with_extra_constraints_covariant_algo():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=None)
    oPs.univariate_slice(extra_constraints=("⟨34⟩+[34]", "⟨34⟩-⟨56⟩", "⟨56⟩+[56]"), seed=0, indepSets=None, algorithm='covariant', verbose=False)

    assert isinstance(oPs("[12]"), sympy.Expr)
    assert sympy.expand(oPs("[12]"), modulus=oPs.field.characteristic).as_poly().degree() == 1


def test_slice_with_extra_constraints_generic_algo():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=None)
    oPs.univariate_slice(extra_constraints=("⟨34⟩+[34]", "⟨34⟩-⟨56⟩", "⟨56⟩+[56]"), seed=0, indepSets=None, algorithm='generic', verbose=False)

    assert isinstance(oPs("[12]"), sympy.Expr)
    assert sympy.expand(oPs("[12]"), modulus=oPs.field.characteristic).as_poly().degree() == 2


def test_bivariate_slice_covariant_algo():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=None)
    oPs.univariate_slice()

    assert isinstance(oPs("⟨12⟩"), sympy.Expr)
    assert isinstance(oPs("[12]"), sympy.Expr)
