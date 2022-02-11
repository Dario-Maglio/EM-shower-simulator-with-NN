""" Train the GAN with the Geant generated dataset and save the model """

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from constants import default_list

# Logger import
from dataset import logData
from make_models import logMod
from class_GAN import logGAN

# Train import
from dataset import debug_data_pull, data_pull
from make_models import make_generator_model, make_discriminator_model
from class_GAN import ConditionalGAN
from unbiased_metrics import shower_depth_lateral_width

#-------------------------------------------------------------------------------

VERBOSE = True

# Path list from this folder
path_list = [os.path.join('..', '..', path) for path in default_list]

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

def global_metrics_real_data():
    train_data = debug_data_pull(path_list, 10000)
    train_images = train_data[0]
    metrics = shower_depth_lateral_width(train_images)
    for el in metrics:
        print(f"{el} = {metrics[el]}")


if __name__=="__main__":
    """Creation and training of the conditional GAN."""
    if VERBOSE :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    logger.info("Start building operations.")
    train_dataset = data_pull(path_list)

    generator = make_generator_model()

    discriminator = make_discriminator_model()

    cond_gan = ConditionalGAN(generator, discriminator)
    logger.info("The cGAN model has been built correctly.")

    global_metrics_real_data()
    history = cond_gan.train(train_dataset, epochs=1, batch=256, wake_up=70)
    np.save(os.path.join("model_results","history.npy"), history)

    # only to remember how to load the dictionary:
    hist = np.load(os.path.join("model_results","history.npy"))
    plt.figure("Evolution of losses per epochs")
    for key in hist:
        plt.plot(history[key], label=key)
    plt.legend()
    plt.show()

    logger.info("The cGAN model has been trained correctly.")

    logger.info("The work is done.")
    logger.handlers.clear()
