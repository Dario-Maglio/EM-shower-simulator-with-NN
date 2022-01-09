""" Train the GAN with the Geant generated dataset and save the model """

# Examples of conditional GANs from which we built our neural network :
# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
# https://keras.io/examples/generative/conditional_gan/

import os
import time
import logging
from sys import exit
from pathlib import Path

import tensorflow as tf

from dataset import DPATH, data_pull, debug_data_pull, debug_shower
from make_models import num_examples, debug_generator, debug_discriminator
from make_models import make_generator_model, make_discriminator_model
from class_GAN import ConditionalGAN

#-------------------------------------------------------------------------------
"""Constant parameters of configuration and definition of global objects."""

# Configuration of the cGAN structure
EPOCHS = 200
L_RATE = 3e-4

BATCH_SIZE = 256
BUFFER_SIZE = 10400

# Define logger and handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger = logging.getLogger("DEBUGLogger")
logger.addHandler(ch)

#-------------------------------------------------------------------------------

def debug(path=DPATH, verbose=False):

    if verbose :
        #Set the logger level to debug
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    try:
        train_data = debug_data_pull(path, num_examples, verbose)
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

    logger.info("Starting training operations.")
    train_dataset = data_pull(path)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

    generator = make_generator_model()

    discriminator = make_discriminator_model()

    cond_gan = ConditionalGAN(discriminator, generator)
    cond_gan.summary()
    cond_gan.compile()
    logger.info("The cGAN model has been compiled correctly.")

    #cond_gan.train(train_dataset, EPOCHS)
    #cond_gan.fit(train_dataset, epochs=EPOCHS)
    # evaluate the model???
    #scores = model.evaluate(X, Y, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


if __name__=="__main__":

    debug(verbose=True)

    train_cgan()

    logger.info("The work is done.")
    logger.handlers.clear()
