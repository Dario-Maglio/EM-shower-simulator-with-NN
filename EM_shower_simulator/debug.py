""" Debug GAN, generator and discriminator and save the models """

import os
import logging
from sys import exit

import matplotlib.pyplot as plt

from dataset import data_path

# Logger import
from dataset import logger as logData
from make_models import logger as logMod
from class_GAN import logger as logGAN

# Debug import
from dataset import debug_data_pull, debug_shower
from make_models import debug_generator, debug_discriminator
from make_models import make_generator_model, make_discriminator_model
from class_GAN import ConditionalGAN

# Creation of the default dataset path
# In the project folder
#DPATH = data_path
# In colab after cloning the repository
#DPATH = os.path.join("EM-shower-simulator-with-NN", data_path)
# In this folder
DPATH = os.path.join("..", data_path)

# Define logger and handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger = logging.getLogger("DEBUGLogger")
logger.addHandler(ch)
logData.addHandler(ch)
logMod.addHandler(ch)
logGAN.addHandler(ch)

#-------------------------------------------------------------------------------

def debug(path=DPATH, verbose=False):
    """Debug subroutines for the training of the cGAN with dataset in path."""
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    try:
        train_data = debug_data_pull(path, 6)
    except AssertionError as e:
        print(f"An error occurred while loading the dataset: \n{e}")
        exit()
    except Exception as e:
        print(f"Error: Invalid path or corrupted file. \n{e}")
        exit()

    if verbose :
        #Execute debug subroutines
        train_images = train_data[0]
        debug_shower(train_images, verbose)
        debug_generator(verbose=verbose)
        debug_discriminator(train_images, verbose)

if __name__=="__main__":

    debug(verbose=True)

    generator = make_generator_model()

    discriminator = make_discriminator_model()

    cond_gan = ConditionalGAN(generator, discriminator)
    logger.info("The cGAN model has been built correctly.")

    cond_gan.summary()
    cond_gan.plot_model()
    logger.info("The cGAN model has been plotted correctly.")

    logger.info("The work is done.")
    logger.handlers.clear()
