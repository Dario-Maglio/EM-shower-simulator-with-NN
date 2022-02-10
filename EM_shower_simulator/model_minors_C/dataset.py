""" Subroutines for the dataset organization and shower plot """

import logging

import numpy as np
import uproot as up
import matplotlib.pyplot as plt
from tensorflow.data import Dataset

#-------------------------------------------------------------------------------

# Basic constants for the dataset configuration
GEOMETRY = (12, 25, 25, 1)
ENERGY_NORM = 6.7404
ENERGY_SCALE = 1000000
BUFFER_SIZE = 10400

# Define logger
logData = logging.getLogger("DataLogger")

#-------------------------------------------------------------------------------

def single_data_pull(path):
    """Organize and reshape the dataset for the cGan training.
    Inputs:
    path = path to a .root file containing the dataset.

    Description:
    Take in input a path to the dataset and return an iterator over events that
    can be used to train the cGAN using the method train.
    """
    logData.info("Loading data from the path.")
    try:
        with up.open(path) as file:
            branches = file["h"].arrays()
    except ValueError as error:
        print("Error: problems opening the file with uproot.")
        raise error
    except OSError as error:
        print("Error: invalid path or corrupted file.")
        raise error

    train_images = np.array(branches["shower"]).astype("float32")
    # Images in log10 scale, empty pixels are set to -1,
    n_event = train_images.shape[0]
    # Normalize the images to [-1, 1] and reshape
    train_images = np.reshape(train_images, (n_event, *GEOMETRY))

    en_labels = np.array(branches["en_in"]).astype("float32") / ENERGY_SCALE
    en_labels = np.transpose(en_labels)
    assert n_event == en_labels.shape[0], "Dataset energy labels compromised!"
    en_labels = np.reshape(en_labels, (n_event, 1))

    # Particle labels are -1, 0, 1 ==> 0, 1, 2 for embedding layer
    pid_labels = np.array(branches["primary"]) + 1
    pid_labels = np.transpose(pid_labels)
    assert n_event == pid_labels.shape[0], "Dataset PID labels compromised!"
    pid_labels = np.reshape(pid_labels, (n_event, 1))

    dataset = Dataset.from_tensor_slices((train_images, en_labels, pid_labels))
    return dataset

def data_pull(path_list):
    """Create the dataset for the cGan training from a path list.
    Inputs:
    path_list = path list to .root files containing data.

    Description:
    Take in input a list of paths to data, call single_data_pull for each
    one, concatenate them and return an iterator over events that can be used
    to train the cGAN.
    """
    for num, path in enumerate(path_list):
        if num==0 :
            dataset = single_data_pull(path)
        else:
            single_dataset = single_data_pull(path)
            dataset = dataset.concatenate(single_dataset)

    dataset = dataset.shuffle(BUFFER_SIZE)
    logData.info(f"Shuffled dataset shape:{dataset.element_spec}")
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
        logData.setLevel(logging.DEBUG)
        logData.info('Logging level set on DEBUG.')
    else:
        logData.setLevel(logging.WARNING)
        logData.info('Logging level set on WARNING.')
    logData.info("Start debugging the dataset loading subroutines.")

    dataset = data_pull(path_list)
    logData.info("Dataset imported.")
    dataset = dataset.batch(num_examples, drop_remainder=True)
    logData.info("Batch created.")

    for batch in dataset:
        train_images, en_labels, pid_labels = batch
        break

    logData.info("Debug of the loading subroutines finished.")
    return [train_images, en_labels, pid_labels]

def debug_shower(data_images, verbose=False):
    """Plot all showers from data_images and return the figure.
    Inputs:
    data_images = vector of images with shape equal to GEOMETRY.
    """
    if verbose :
        logData.setLevel(logging.DEBUG)
        logData.info('Logging level set on DEBUG.')
    else:
        logData.setLevel(logging.WARNING)
        logData.info('Logging level set on WARNING.')

    logData.info("Start debug data_images features.")
    logData.info(f"Shape of data_images: {data_images.shape}")
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
    logData.info("Debug of data_images features finished.")
    return fig
