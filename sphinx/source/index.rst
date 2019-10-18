.. seampy documentation master file, created by
   sphinx-quickstart on Tue Sep 24 12:11:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lips
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

lips (Lorentz Invariant Phase Space) is an object-oriented phase space generator written entirely in python.


Installation
=================

Installation is easy with pip::

  pip install lips

alternatively the package can be cloned from github at https://github.com/GDeLaurentis/lips.


Quick start
=================

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
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
