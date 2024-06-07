Lorentz Invariant Phase Space
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

lips is an object-oriented phase space generator for theory computations.


Installation
------------

Installation is easy with pip::

  pip install lips

alternatively the package can be cloned from github at https://github.com/GDeLaurentis/lips.


Quick start
-----------

To generate a phase space point follow this very simple example:

  .. code-block:: python
		  :linenos:
		   
		     import lips
		     oParticles = lips.Particles(6)

this is a six-point phase space point, with complex momenta, satisfying both momentum conservation and massless on-shell relations.
For a real phase space point pass an optional argument:

  .. code-block:: python
		  :linenos:
		   
		     import lips
		     oParticles = lips.Particles(6, real_momenta=True)

Particle objects have several attributes, corresponding to the different representations of the Lorentz group. For a full list check :ref:`modindex`.
Changing an attribute, for instance oParticles[1].four_mom, automatically updates all other representations, such as the rank two spinor oParticles[1].r2_sp.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. Hidden TOCs

.. toctree::
   :caption: Modules Documentation
   :maxdepth: 2

   modules
