import pytest

from lips.algebraic_geometry.covariant_ideal import LipsIdeal

from shutil import which

singular_found = True if which('Singular') is not None else False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.skipif(not singular_found, reason="singular not found")
def test_LipsIdeal_membership():
    assert "⟨1|3⟩" in LipsIdeal(5, ("⟨1|3⟩", "⟨3|1+5|3]", ))
    assert "⟨2|4⟩" not in LipsIdeal(5, ("⟨1|3⟩", "⟨3|1+5|3]", ))
    assert "⟨3|1+5|3]" in LipsIdeal(5, ("⟨1|3⟩", "⟨3|5⟩", ))


@pytest.mark.skipif(not singular_found, reason="singular not found")
def test_LipsIdeal_intersection():
    I1 = LipsIdeal(5, ('⟨1|2⟩', '⟨1|3⟩'))
    P1 = LipsIdeal(5, ('⟨1|2⟩', '⟨1|3⟩', '⟨2|3⟩', '[4|5]'))
    P2 = LipsIdeal(5, ('⟨1|', ))
    P3 = LipsIdeal(5, ('⟨1|2⟩', '⟨1|3⟩', '⟨1|4⟩', '⟨1|5⟩', '⟨2|3⟩', '⟨2|4⟩', '⟨2|5⟩', '⟨3|4⟩', '⟨3|5⟩', '⟨4|5⟩'))
    assert I1 == P1 & P2 & P3


@pytest.mark.skipif(not singular_found, reason="singular not found")
def test_LipsIdeal_quotient():
    I1 = LipsIdeal(5, ('⟨1|2⟩', '⟨1|3⟩'))
    P1 = LipsIdeal(5, ('⟨1|2⟩', '⟨1|3⟩', '⟨2|3⟩', '[4|5]'))
    P2 = LipsIdeal(5, ('⟨1|', ))
    P3 = LipsIdeal(5, ('⟨1|2⟩', '⟨1|3⟩', '⟨1|4⟩', '⟨1|5⟩', '⟨2|3⟩', '⟨2|4⟩', '⟨2|5⟩', '⟨3|4⟩', '⟨3|5⟩', '⟨4|5⟩'))
    assert I1 / P1 / P2 == P3


@pytest.mark.skipif(not singular_found, reason="singular not found")
def test_symmetry_image_commutes_with_invariant_slice():
    P1 = LipsIdeal(6, ('⟨1|2⟩', '⟨1|3⟩', '⟨2|3⟩'))
    P1permuted = P1('126345', False)
    iP1permuted1 = P1permuted.invariant_slice()
    iP1 = P1.invariant_slice()
    iP1permuted2 = iP1('126345', False)
    assert iP1permuted1 == iP1permuted2


def test_primality_test():
    I = LipsIdeal(6, ('⟨1|2⟩',
                      '-1⟨2|3⟩⟨2|4⟩[1|4][2|3]-1⟨2|4⟩²[1|4][2|4]+1⟨2|3⟩⟨3|4⟩[1|3][3|4]',
                      '-1⟨1|3⟩⟨2|4⟩[1|4][2|3]-1⟨1|4⟩⟨2|4⟩[1|4][2|4]+1⟨1|3⟩⟨3|4⟩[1|3][3|4]'))
    I.ring.variables = I.ring.variables[::-1]
    I.test_primality(verbose=True, seminumerical_dim_computation=True, astuple=False, timeout_fpoly=5, nbr_points=0,
                     projection_number=(1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1))
