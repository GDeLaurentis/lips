# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from lips import Particles
from lips.algebraic_geometry.ideal import LipsIdeal


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_LipsIdeal_membership():
    assert "⟨1|3⟩" in LipsIdeal(Particles(5), ("⟨1|3⟩", "⟨3|(1+5)|3]", ))
    assert "⟨2|4⟩" not in LipsIdeal(Particles(5), ("⟨1|3⟩", "⟨3|(1+5)|3]", ))
    assert "⟨3|(1+5)|3]" in LipsIdeal(Particles(5), ("⟨1|3⟩", "⟨3|5⟩", ))
