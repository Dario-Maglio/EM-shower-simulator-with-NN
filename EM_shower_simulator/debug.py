""" Debug GAN, generator and discriminator and save the models """

import os
import sys
import logging

import matplotlib.pyplot as plt

from constants import default_list

# Logger import
from dataset import logger as logData
from make_models import logger as logMod
from class_GAN import logger as logGAN

# Debug import
from dataset import debug_data_pull, debug_shower
from make_models import debug_generator, debug_discriminator
from make_models import make_generator_model, make_discriminator_model
from class_GAN import test_noise, ConditionalGAN, compute_energy

#-------------------------------------------------------------------------------

VERBOSE = True

# Path list from this folder
path_list = [os.path.join('..', path) for path in default_list]

# Examples to show
EXAMPLES = 5

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

def debug(path_list, num_examples=EXAMPLES, verbose=False):
    """Debug subroutines for the training of the cGAN with dataset in path."""
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    try:
        train_data = debug_data_pull(path_list, num_examples, verbose=verbose)
    except AssertionError as error:
        print(f"An error occurred while loading the dataset: \n{error}")
        sys.exit()

    #Execute debug subroutines
    train_images = train_data[0]
    debug_shower(train_images, verbose)
    debug_generator(test_noise, verbose=verbose)
    debug_discriminator(train_images, verbose)

def debug_cgan(gan, path_list, num_examples=EXAMPLES):
    """Debug of the cGAN methods."""
    logger.info("Testing the cGAN methods on noise and real samples.")
    noise = gan.generate_noise(num_examples)
    gan.generate_and_save_images(noise)

    gener, discr = gan.evaluate()

    # Fake showers
    predictions = gener(noise, training=False)
    decisions = discr(predictions, training=False)
    energy = compute_energy(predictions)

    k = 0
    num_examples = predictions.shape[0]
    fig = plt.figure("Fake generated showers", figsize=(20,10))
    for i in range(num_examples):
        print(f"Example {i+1}\n"
             +f"Primary particle = {int(noise[2][i][0])}\t"
             +f"Predicted particle = {decisions[2][i][0]}\n"
             +f"Initial energy = {noise[1][i][0]}\t"
             +f"Generated energy = {energies[i][0]}\t"
             +f"Predicted energy = {decisions[1][i][0]}\t"
             +f"Decision = {decisions[0][i][0]}\n\n")
        for j in range(predictions.shape[1]):
            k=k+1
            plt.subplot(num_examples, predictions.shape[1], k)
            plt.imshow(predictions[i,j,:,:,0]) #, cmap="gray")
            plt.axis("off")
    plt.show()
    
    # True showers
    predictions = debug_data_pull(path_list, num_examples)
    images = predictions[0]
    decisions = discr(images, training=False)
    energy = compute_energy(images)

    k = 0
    num_examples = predictions.shape[0]
    fig = plt.figure("Real generated showers", figsize=(20,10))
    for i in range(num_examples):
        print(f"Example {i+1}\n"
             +f"Primary particle = {int(noise[2][i][0])}\t"
             +f"Predicted particle = {decisions[2][i][0]}\n"
             +f"Initial energy = {noise[1][i][0]}\t"
             +f"Generated energy = {energies[i][0]}\t"
             +f"Predicted energy = {decisions[1][i][0]}\t"
             +f"Decision = {decisions[0][i][0]}\n\n")
        for j in range(predictions.shape[1]):
            k=k+1
            plt.subplot(num_examples, predictions.shape[1], k)
            plt.imshow(predictions[i,j,:,:,0]) #, cmap="gray")
            plt.axis("off")
    plt.show()

    logger.info("Debug of the cGAN methods finished.")


if __name__=="__main__":

    debug(path_list, verbose=VERBOSE)

    generator = make_generator_model()

    discriminator = make_discriminator_model()

    cond_gan = ConditionalGAN(generator, discriminator)
    logger.info("The cGAN model has been built correctly.")

    cond_gan.summary()
    cond_gan.plot_model()
    logger.info("The cGAN model has been plotted correctly.")

    debug_cgan(cond_gan, path_list)

    logger.info("The work is done.")
    logger.handlers.clear()
