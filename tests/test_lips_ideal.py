# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from lips.algebraic_geometry.ideal import LipsIdeal
from lips.algebraic_geometry.tools import singular_found


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.skipif(not singular_found, reason="singular not found")
def test_LipsIdeal_membership():
    assert "⟨1|3⟩" in LipsIdeal(5, ("⟨1|3⟩", "⟨3|(1+5)|3]", ))
    assert "⟨2|4⟩" not in LipsIdeal(5, ("⟨1|3⟩", "⟨3|(1+5)|3]", ))
    assert "⟨3|(1+5)|3]" in LipsIdeal(5, ("⟨1|3⟩", "⟨3|5⟩", ))
