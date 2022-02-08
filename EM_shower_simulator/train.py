""" Train the GAN with the Geant generated dataset and save the model """

import os
import logging

import matplotlib.pyplot as plt

from constants import default_list

# Logger import
from dataset import logger as logData
from make_models import logger as logMod
from class_GAN import logger as logGAN

# Train import
from dataset import data_pull
from make_models import make_generator_model, make_discriminator_model
from class_GAN import ConditionalGAN

#-------------------------------------------------------------------------------

VERBOSE = True

# Path list from this folder
for index, path in enumerate(default_list):
    default_list[index] = os.path.join('..', path)

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

def train_cgan(cond_gan, train_dataset):
    history = cond_gan.train(train_dataset, epochs=5, batch=256, wake_up=70)
    #history = cond_gan.fit(train_dataset, epochs=3, batch=2048)

    plt.figure("Evolution of losses per epochs")
    for key in history:
        plt.plot(history[key], label=key)
    plt.show()
    logger.info("The cGAN model has been trained correctly.")

if __name__=="__main__":
    """Creation and training of the conditional GAN."""
    if VERBOSE :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    logger.info("Start building operations.")
    train_dataset = data_pull(default_list)

    generator = make_generator_model()

    discriminator = make_discriminator_model()

    cond_gan = ConditionalGAN(generator, discriminator)
    logger.info("The cGAN model has been built correctly.")

    train_cgan(cond_gan, train_dataset)

    logger.info("The work is done.")
    logger.handlers.clear()
