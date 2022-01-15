""" Train the GAN with the Geant generated dataset and save the model """

import os
import logging
from sys import exit

import matplotlib.pyplot as plt

from dataset import data_path

# Logger import
from dataset import logger as logData
from make_models_C import logger as logMod
from class_GAN import logger as logGAN

# Train import
from dataset import data_pull
from make_models_C import make_generator_model, make_discriminator_model
from class_GAN import ConditionalGAN


# Creation of the default dataset path
# In the project folder
#path = data_path
# In colab after cloning the repository
#path = os.path.join("EM-shower-simulator-with-NN", data_path)
# In this folder
path = os.path.join("..", data_path)

verbose = True

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
    history = cond_gan.train(train_dataset, epochs=100, batch=64, wake_up=10)
    #history = cond_gan.fit(train_dataset, epochs=3, batch=2048)

    plt.figure("Evolution of losses per epochs")
    plt.plot(history["gener_loss"], label="gener_loss")
    plt.plot(history["discr_loss"], label="discr_loss")
    plt.plot(history["energ_loss"], label="energ_loss")
    plt.plot(history["label_loss"], label="label_loss")
    plt.show()
    logger.info("The cGAN model has been trained correctly.")

if __name__=="__main__":
    """Creation and training of the conditional GAN."""
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    logger.info("Start building operations.")
    train_dataset = data_pull(path)

    generator = make_generator_model()

    discriminator = make_discriminator_model()

    cond_gan = ConditionalGAN(generator, discriminator)
    logger.info("The cGAN model has been built correctly.")

    train_cgan(cond_gan, train_dataset)

    logger.info("The work is done.")
    logger.handlers.clear()
