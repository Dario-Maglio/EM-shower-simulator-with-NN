EM shower simulator with Neural Network
=======================================


.. image:: https://app.travis-ci.com/Dario-Caf/EM-shower-simulator-with-NN.svg?branch=main
   :target: https://app.travis-ci.com/Dario-Caf/EM-shower-simulator-with-NN

.. image:: https://readthedocs.org/projects/em-shower-simulator-with-nn/badge/?version=latest
   :target: https://em-shower-simulator-with-nn.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status



Recent developments in calorimeter physics are leading to a new paradigm in
calorimeters' design. The aim is to reconstruct the whole spatial distribution
of the shower instead of extracting information only about the energy deposition
inside calorimeter cells. Many collaborations are designing highly-segmented
calorimeters (for instance, CMS High-Granularity CAL and ALICE FoCAL) to reach
this goal. In this project, our goal is to build a neural network to simulate
electromagnetic shower energy deposition inside a toy segmented calorimeter.
Taking inspiration from recent works in this field, like the CaloGAN network,
we have built a conditional GAN with auxiliary conditions based on total energy
deposition and particle IDs. The neural network was trained with a dataset
created on purpose with the Geant4 simulation toolkit. Simulations involve
electrons, positrons and photons with energies from 1 to 30 GeV that strike on a
CsI calorimeter with 12 layers and 12x12 cells. The results are further analyzed
with ROOT to evaluate GAN's performances. Due to time, dataset and hardware
constraints, this project must be considered an exploratory work whose results
can even be improved with more resources.

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

In python you can import the module simply writing

.. code-block:: python

   >>> import EM_shower_simulator as EM

then you can run the simulation module as

.. code-block:: python

   >>> EM.simulate([10.0, 1], verbose=0)
   simulating event with features [10.0, 1.0]
   0

This returns the error code and shows the resulting plot of the simulation.

You can also run the simulation from command line, writing:

.. code-block:: bash

   $ simulate-EM-shower --features 10. 1

This command allows the user to generate a shower simulation from command line.
Furthermore, it allows to pass the shower's features as float arguments in the
order: energy momentum angle
Use --help for more information about the arguments:

.. code-block:: bash

   $ simulate-EM-shower --help
   usage: simulate-EM-shower [-h] [-v] [-f feature [feature ...]]
   ...

It gives an error if a different input size or type is passed.
