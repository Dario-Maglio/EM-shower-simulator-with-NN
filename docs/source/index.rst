.. Electromagnetic shower simulator with NN documentation master file, created by
   sphinx-quickstart on Tue Dec  7 16:11:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../../README.md

Installation
------------

After cloning the project directory, you can install the package in development
mode writing from bash

.. code-block:: bash

   pip install -e .


Usage
-----

If no arguments are passed, the simulation will use random features.
(not implemented yet -> actually default parameter in use)

In a python environment you can use the module simply writing

.. code-block:: python

   >>> from EM_shower_simulator import simulate_shower as simsh
   >>> simsh([1., 3., 5.], verbose=0)
   #return the error code and show the result
   0

You can also execute the bash script

.. code-block:: bash

   simulate_EM_shower --features 1. 3. 5.

Use --help for more informatioin

API
---

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
