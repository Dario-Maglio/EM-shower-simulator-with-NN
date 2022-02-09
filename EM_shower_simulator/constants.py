import os

#-------------------------------------------------------------------------------

# Creation of the default dataset path
FILE_LIST = ["data_MVA_24pixel_parte1.root", "data_MVA_24pixel_parte2.root"]
# FILE_LIST = ["data_MVA_normalized.root"]
default_list = []
for file in FILE_LIST:
    default_list.append(os.path.join("dataset", "filtered_data", file))

# Checkpoint path
CHECKP = os.path.join('EM_shower_simulator','training_checkpoints','checkpoint')

#-------------------------------------------------------------------------------

ENERGY_NORM = 6.7404
ENERGY_SCALE = 1000000
GEOMETRY = (12, 25, 25, 1)

N_PID = 3                               # number of pid classes
N_ENER = 30 + 1                         # number of en classes
PARAM_EN = 0.01                         # parameter in energy losses computation
NOISE_DIM = 2048
BUFFER_SIZE = 10400

MBSTD_GROUP_SIZE = 8                    #minibatch dimension

#-------------------------------------------------------------------------------
