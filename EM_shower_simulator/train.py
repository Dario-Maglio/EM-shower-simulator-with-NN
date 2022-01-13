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

def train_cgan(path=DPATH, verbose=False):
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

    cond_gan.summary()
    cond_gan.plot_model()
    logger.info("The cGAN model has been plotted correctly.")

    history = cond_gan.train(train_dataset, epochs=100)
    #history = cond_gan.fit(train_dataset, epochs=3, batch=2048)

    for key in history.keys():
        print(key)

    plt.figure("Evolution of losses per epochs")
    plt.plot(history["gener_loss"])
    plt.plot(history["discr_loss"])
    plt.plot(history["energ_loss"])
    plt.plot(history["label_loss"])
    plt.show()
    logger.info("The cGAN model has been trained correctly.")

    noise = cond_gan.generate_noise(4)
    cond_gan.generate_and_save_images(noise)
    logger.info("The cGAN model has been tested correctly.")

    gener, discr = cond_gan.evaluate()
    predictions = discr(train_dataset[0][0:2])
    print(predictions)

if __name__=="__main__":

    train_cgan(verbose=True)

    logger.info("The work is done.")
    logger.handlers.clear()
