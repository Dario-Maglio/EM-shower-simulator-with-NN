Titolino folder README
----------------------

Structure of data_MVA.root:

*  ROOT file containing a tree named "h";

Branches inside "h":

* "evt"     : event number;

* "primary" : identity of the particle generating the shower (photon=0, electron=1, positron=-1)

* "en_in"   : energy of the particle generating the shower (in keV);

* "theta"   : theta angle of incidence of the primary particle;

* "phi"     : phi angle of incidence of the primary particle;

* "shower"  : 4D array of dimensions (layers=12,pixel_x=12,pixely=12,1) containing the log10-values of the energy deposited by the primary particle in each pixel of the detector.


The structure of the branch "shower" is specific for our simulation geometry,
defined in "dataset/data_config_Geant4/geometria.tg".

If you want to change the geometry of the simulation, please see
"dataset/data_config_Geant4/README.rst" and then modify "analisi.C" file in this
folder.
