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

from syngular import Field                       # noqa

from .particle import Particle                   # noqa
from .particles import Particles                 # noqa
from .tools import myException, ldot, flatten    # noqa
from .invariants import Invariants               # noqa

spinor_convention = 'symmetric'  # or 'asymmetric'
