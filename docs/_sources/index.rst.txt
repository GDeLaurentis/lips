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

To get started generating phase space points open an interactive python session and follow this simple example:

  .. code-block:: python
		  :linenos:
		   
		     import lips
		     oParticles = lips.Particles(6)
		     oParticles.fix_mom_cons()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
