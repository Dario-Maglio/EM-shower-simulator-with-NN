import os

#-------------------------------------------------------------------------------

# Creation of the default dataset path
FILE_LIST = ["data_MVA_24pixel_parte1.root", "data_MVA_24pixel_parte2.root"]
# FILE_LIST = ["data_MVA_normalized.root"]
default_list = []
for file in FILE_LIST:
    default_list.append(os.path.join("dataset", "filtered_data", file))

# Checkpoint path
CHECKP = os.path.join('EM_shower_simulator','checkpoints','checkpoint')

#-------------------------------------------------------------------------------

N_PID = 3
NOISE_DIM = 1024
ENERGY_NORM = 6.7404
ENERGY_SCALE = 1000000
GEOMETRY = (12, 25, 25, 1)

#-------------------------------------------------------------------------------
