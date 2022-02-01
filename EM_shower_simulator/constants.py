import os

# 12x12x12 GEOMETRY
# GEOMETRY = (12, 12, 12, 1)
#
data_path = os.path.join("dataset","filtered_data","data_MVA_normalized.root")
DPATH = os.path.join("..", data_path)
# PATH_LIST = [ DPATH ]


#------------------------------------------------------------------------------#

# 12x25x25 GEOMETRY
GEOMETRY = (12, 25, 25, 1)

data_path_1 = os.path.join("dataset","filtered_data","data_MVA_24pixel_parte1.root")
DPATH_1 = os.path.join("..", data_path_1)

data_path_2 = os.path.join("dataset","filtered_data","data_MVA_24pixel_parte2.root")
DPATH_2 = os.path.join("..", data_path_2)

PATH_LIST = [ DPATH_1, DPATH_2 ]

#------------------------------------------------------------------------------#

ENERGY_NORM = 6.7404
ENERGY_SCALE = 1000000

N_PID = 3                               # number of pid classes
N_ENER = 30 + 1                         # number of en classes
PARAM_EN = 0.01                         # parameter in energy losses computation
NOISE_DIM = 1024
BUFFER_SIZE = 10400

MBSTD_GROUP_SIZE = 8                    #minibatch dimension
