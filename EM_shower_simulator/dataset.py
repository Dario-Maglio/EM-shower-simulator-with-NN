"""Subroutines for the dataset organization and shower plot."""

import os
import logging

import numpy as np
import uproot as up
import matplotlib.pyplot as plt
from tensorflow.data import Dataset

#-------------------------------------------------------------------------------

# Creation of the default dataset path and configuration of the dataset structure
# from constants import *
data_path = os.path.join("dataset","filtered_data","data_MVA_normalized.root")
DPATH = os.path.join("..", data_path)
# PATH_LIST = [ DPATH ]

data_path_1 = os.path.join("dataset","filtered_data","data_MVA_24pixel_parte1.root")
DPATH_1 = os.path.join("..", data_path_1)

data_path_2 = os.path.join("dataset","filtered_data","data_MVA_24pixel_parte2.root")
DPATH_2 = os.path.join("..", data_path_2)

PATH_LIST = [ DPATH_1, DPATH_2 ]


# Define logger
logger = logging.getLogger("DataLogger")

#-------------------------------------------------------------------------------

def single_data_pull(path, verbose=False):
    """Organize and reshape the dataset for the cGan training.
    Inputs:
    path = path to a .root file containing the dataset.

    Description:
    Take in input a path to the dataset and return an iterator over events that
    can be used to train the cGAN using the method train.
    """
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    logger.info("Loading the training dataset.")
    try:
       with up.open(path) as file:
           branches = file["h"].arrays()
    except Exception as e:
       print("Error: Invalid path or corrupted file.")
       raise e

    train_images = np.array(branches["shower"]).astype("float32")
    # Images in log10 scale, empty pixels are set to -1,
    N_EVENT = train_images.shape[0]
    # Normalize the images to [-1, 1] and reshape
    train_images = np.reshape(train_images, (N_EVENT, *GEOMETRY))

    en_labels = np.array(branches["en_in"]).astype("float32") / ENERGY_SCALE
    en_labels = np.transpose(en_labels)
    assert N_EVENT == en_labels.shape[0], "Dataset energy labels compromised!"
    en_labels = np.reshape(en_labels, (N_EVENT, 1))

    # Particle labels are -1, 0, 1 ==> 0, 1, 2 for embedding layer
    pid_labels = np.array(branches["primary"]) + 1
    pid_labels = np.transpose(pid_labels)
    assert N_EVENT == pid_labels.shape[0], "Dataset PID labels compromised!"
    pid_labels = np.reshape(pid_labels, (N_EVENT, 1))

    return [train_images, en_labels, pid_labels]

def data_pull(path_list):
    """ Merges data from multiple paths
    """
    for num, path in enumerate(path_list):
        train_images_path, en_labels_path, pid_labels_path = single_data_pull(path)
        if len(path_list)==1 or num==0 :
            train_images = train_images_path
            en_labels    = en_labels_path
            pid_labels   = pid_labels_path
        else :
            train_images = np.concatenate([train_images, train_images_path], axis=0)
            en_labels = np.concatenate([en_labels, en_labels_path], axis=0)
            pid_labels = np.concatenate([pid_labels,pid_labels_path],axis=0)

    dataset = Dataset.from_tensor_slices((train_images, en_labels, pid_labels))
    dataset = dataset.shuffle(BUFFER_SIZE)
    return dataset

def debug_data_pull(path_list, num_examples=1, verbose=False):
    """Import data images from the dataset and test shapes.
    Inputs:
    path = path to a .root file containing the dataset;
    num_examples = number of random shower to plot.

    Description:
    Take in input a path to the dataset and return num_examples of events from
    the dataset that can be plotted with debug_shower.
    """
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    logger.info("Start debugging the dataset loading subroutines.")

    for num, path in enumerate(path_list):
        train_images_path, en_labels_path, pid_labels_path = single_data_pull(path)
        if len(path_list)==1 or num == 0:
            train_images = train_images_path
            en_labels    = en_labels_path
            pid_labels   = pid_labels_path
        else :
            train_images = np.concatenate([train_images, train_images_path], axis=0)
            en_labels = np.concatenate([en_labels, en_labels_path], axis=0)
            pid_labels = np.concatenate([pid_labels,pid_labels_path],axis=0)

    train_images = train_images[0:num_examples]
    en_labels = en_labels[0:num_examples]
    pid_labels = pid_labels[0:num_examples]

    logger.info("Debug of the loading subroutines finished.")
    return [train_images, en_labels, pid_labels]

def debug_shower(data_images, verbose=False):
    """Plot all showers from data_images and return the figure.
    Inputs:
    data_images = vector of images with shape equal to GEOMETRY.
    """
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    logger.info("Start debug data_images features.")
    logger.info(f"Shape of data_images: {data_images.shape}")
    k=0
    fig = plt.figure("Generated showers", figsize=(20,10))
    num_examples = data_images.shape[0]
    for i in range(num_examples):
       for j in range(GEOMETRY[0]):
          k=k+1
          plt.subplot(num_examples, GEOMETRY[0], k)
          plt.imshow(data_images[i,j,:,:,0]) #, cmap="gray")
          plt.axis("off")
    plt.show()
    logger.info("Debug of data_images features finished.")
    return fig
