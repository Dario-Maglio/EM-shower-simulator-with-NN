Geant4 simulation configuration
-------------------------------

Geant4 simulation is executed by "gears.exe" programm, which is a precompiled
version of Geant4, compatible with Windows and Linux. For more informations and
installation guide:

https://github.com/jintonic/gears

In order to modify the simulation:

* Manage simulation geometry in "geometria.tg" and in "file.py" ; then run:

.. code-block:: bash

   python file.py

* Manage particle source and energy in "sorgente.mac" ;
* Manage overall settings (output file name, number of events to generate, ...) ;
* To launch the simulation, type in cmd (in Windows) opened in this folder :

.. code-block:: bash

   gears.exe simulazione.mac


Output file shower.root from Geat4 simulation contains:

* Main tree "t" ;

Branches inside the tree:

* Track id (trk in short) ;
* Step number, starting from 0 (stp in short) ;
* Detector volume copy number (vlm in short) ;
* Process id (pro in short) ;
* Particle id (pdg in short) ;
* Track id of the parent particle (pid in short) ;
* Local position xx [mm] (origin: center of the volume) ;
* Local position yy [mm] (origin: center of the volume) ;
* Local position zz [mm] (origin: center of the volume) ;
* Local time dt [ns] (time elapsed from previous step point) ;
* Energy deposited [keV] (de in short) ;
* Step length [mm] (dl in short) ;
* Trajectory length [mm] (l in short) ;
* Global position x [mm] (origin: center of the world) ;
* Global position y [mm] (origin: center of the world) ;
* Global position z [mm] (origin: center of the world) ;
* Global time t [ns] (time elapsed from the beginning of event) ;
* Kinetic energy of the particle [keV] (k in short) ;
* Momentum of the particle [keV] (p in short) ;
* Charges (q in short) .
* Total energy deposited in a sensitive volume [keV] (et in short)

This file is then processed with "dataset/filtered_data/MVA_processing.C", in order to
create a .root file with an appropriate format for the multivariate analysis.
