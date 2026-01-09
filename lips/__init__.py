"""
Defines tools for phase space manipulations.
Particles objects are base one lists of Particle objects.\n
Particles objects allow to:\n
1) Compute spinor strings, through .compute;\n
2) Construct single collinear limits, through .set;\n
3) Construct double collinear limits, through .set_pair.\n
.. code-block:: python
   :linenos:

   oParticles = Particles(multiplicity)
   oParticles.randomise_all()
   oParticles.fix_mom_cons()
   oParticles.compute(spinor_string)
   oParticles.set(spinor_string, small_value)
   oParticles.set_pair(spinor_string_1, small_value_1, spinor_string_2, small_value_2)
"""

from .version import __version__
from .particle import Particle
from .particles import Particles
from .tools import myException, ldot
from .invariants import Invariants

spinor_convention = 'symmetric'  # or 'asymmetric'

__all__ = [
    "__version__",
    "Particle",
    "Particles",
    "myException",
    "ldot",
    "Invariants",
]


# Back-compatibility - to be removed

import warnings  # noqa


def __getattr__(name):
    if name in {"flatten", }:
        warnings.warn(
            f"lips.{name} is deprecated and will be removed in a future release; "
            f"use pycoretools.iterables.{name} (or: from pycoretools import {name}).",
            FutureWarning,
            stacklevel=2,
        )
        from pycoretools import iterables
        return getattr(iterables, name)
    if name in {"Field", }:
        warnings.warn(
            "lips.Field is deprecated and will be removed in a future release; "
            "use syngular.Field instead.",
            FutureWarning,
            stacklevel=2,
        )
        from syngular import Field
        return Field

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["flatten", "Field", ])
