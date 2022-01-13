""" Train the GAN with the Geant generated dataset and save the model """

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

# Train import
from dataset import data_pull
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
        debug_discriminator(train_data, verbose=verbose)

def train_cgan(path=DPATH, verbose=False):
    """Creation and training of the conditional GAN."""
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    logger.info("Starting training operations.")
    train_dataset = data_pull(path)

    generator = make_generator_model()

    discriminator = make_discriminator_model()

    cond_gan = ConditionalGAN(generator, discriminator)
    cond_gan.compile()
    logger.info("The cGAN model has been compiled correctly.")

    cond_gan.summary()
    cond_gan.plot_model()

    history = cond_gan.train(train_dataset, epochs=1, batch=64)
    #history = cond_gan.fit(train_dataset, epochs=3, batch=2048)

    plt.figure("Evolution of losses per epochs")
    plt.plot(history.history["gener_loss"])
    plt.plot(history.history["discr_loss"])
    plt.show()

    for key in history.history.keys():
        print(key)

    #file_name = "cGAN.h5"
    #save_path = "model_saves"
    #if not os.path.isdir(save_path):
    #   os.makedirs(save_path)
    #cond_gan.save(os.path.join(save_path, file_name))

if __name__=="__main__":

    debug(verbose=True)

    train_cgan(verbose=True)

    logger.info("The work is done.")
    logger.handlers.clear()
