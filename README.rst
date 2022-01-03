EM shower simulator with Neural Network
=======================================

.. image:: https://app.travis-ci.com/Dario-Caf/EM-shower-simulator-with-NN.svg?branch=main
   :target: https://app.travis-ci.com/Dario-Caf/EM-shower-simulator-with-NN

.. image:: https://readthedocs.org/projects/em-shower-simulator-with-nn/badge/?version=latest
   :target: https://em-shower-simulator-with-nn.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

In this project we are involved in the design of a neural network capable of
generating Monte Carlo simulations of electromagnetic swarms. The reference
data and on which the network is trained were generated through a Monte Carlo
simulation with Geant4, with an ideal detector.

Installation
------------

Installing from local source in Development Mode, i.e. in such a way that the
project appears to be installed, but yet is still editable from the src tree:

.. code-block:: bash

   $ git clone https://github.com/Dario-Caf/EM-shower-simulator-with-NN.git
   $ cd EM-shower-simulator-with-NN
   $ python3 -m pip install -e .

All of the package's informations are stored in setup.cfg and are passed as
arguments to setuptools.setup() when it is executed through pip install.

Usage
-----

If no arguments are passed, the simulation will use random features as input.
(not implemented yet -> actually default parameter in use)

In python you can import the module simply writing

.. code-block:: python

   >>> import EM_shower_simulator as EM

then you can run the simulation module as

.. code-block:: python

   >>> EM.simulate([1., 3., 5.], verbose=0)
   simulating event with features [1., 3., 5. ]
   0

This returns the error code and shows the resulting plot of the simulation.

You can also run the simulation from command line, writing:

.. code-block:: bash

   $ simulate-EM-shower --features 1. 3. 5.

This command allows the user to generate a shower simulation from command line.
Furthermore, it allows to pass the shower's features as float arguments in the
order: energy momentum angle
Use --help for more information about the arguments:

.. code-block:: bash

   $ simulate-EM-shower --help
   usage: simulate-EM-shower [-h] [-v] [-f feature [feature ...]]
   ...

It gives an error if a different input size or type is passed.
