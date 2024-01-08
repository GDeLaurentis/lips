# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
