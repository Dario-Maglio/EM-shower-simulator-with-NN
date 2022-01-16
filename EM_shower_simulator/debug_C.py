""" Debug GAN, generator and discriminator and save the models """

import os
import logging
from sys import exit

import matplotlib.pyplot as plt

from dataset import data_path_1, data_path_2

# Logger import
from dataset import logger as logData
from make_models_C import logger as logMod
from class_GAN import logger as logGAN

# Debug import
from dataset import debug_data_pull, debug_shower
from make_models_C import debug_generator, debug_discriminator
from make_models_C import make_generator_model, make_discriminator_model
from class_GAN import test_noise, ConditionalGAN, compute_energy

# Creation of the default dataset path
# In the project folder
#DPATH = data_path
# In colab after cloning the repository
#DPATH = os.path.join("EM-shower-simulator-with-NN", data_path)
# In this folder
DPATH_1 = os.path.join("..", data_path_1)
DPATH_2 = os.path.join("..", data_path_2)

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

def debug(path_1=DPATH_1, path_2=DPATH_2, num_examples=5, verbose=False):
    """Debug subroutines for the training of the cGAN with dataset in path."""
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    try:
        train_data = debug_data_pull(path_1, path_2, num_examples)
    except AssertionError as e:
        print(f"An error occurred while loading the dataset: \n{e}")
        exit()
    except Exception as e:
        print(f"Error: Invalid path or corrupted file. \n{e}")
        exit()

    #Execute debug subroutines
    train_images = train_data[0]
    debug_shower(train_images, verbose)
    debug_generator(test_noise, verbose=verbose)
    debug_discriminator(train_images, verbose)

def debug_cgan(cond_gan, path_1=DPATH_1, path_2=DPATH_2, num_examples=5):
    logger.info("Testing the cGAN methods on noise and real samples.")
    gener, discr = cond_gan.evaluate()
    noise = cond_gan.generate_noise(num_examples)

    # Fake showers
    predictions = gener(noise, training=False)
    decisions = discr(predictions, training=False)
    energy = compute_energy(predictions)

    k=0
    fig = plt.figure("Fake generated showers", figsize=(20,10))
    num_examples = predictions.shape[0]
    for i in range(num_examples):
       for j in range(predictions.shape[1]):
          k=k+1
          plt.subplot(num_examples, predictions.shape[1], k)
          plt.imshow(predictions[i,j,:,:,0]) #, cmap="gray")
          plt.axis("off")

    for example in range(len(noise[0])):
        print(f"{example+1}) Primary particle = {int(noise[2][example][0])}"
             +f"\nInitial energy = {noise[1][example][0]}   "
             +f"Generated energy = {energy[example][0]}   "
             +f"Predicted energy = {decisions[1][example][0]}   "
             +f"Decision = {decisions[0][example][0]}")

    # True showers
    predictions = debug_data_pull(path_1, path_2, num_examples)
    images = predictions[0]
    decisions = discr(images, training=False)
    energy = compute_energy(images)

    k=0
    plt.figure("Real generated showers", figsize=(20,10))
    num_examples = images.shape[0]
    for i in range(num_examples):
       for j in range(images.shape[1]):
          k=k+1
          plt.subplot(num_examples, images.shape[1], k)
          plt.imshow(images[i,j,:,:,0]) #, cmap="gray")
          plt.axis("off")

    for example in range(num_examples):
        print(f"{example+1}) Primary particle = {int(predictions[2][example][0])}"
             +f"\nInitial energy = {predictions[1][example][0]}   "
             +f"Generated energy = {energy[example][0]}   "
             +f"Predicted energy = {decisions[1][example][0]}   "
             +f"Decision = {decisions[0][example][0]}")

    plt.show()
    logger.info("Debug of the cGAN methods finished.")


if __name__=="__main__":

    debug(verbose=True)

    generator = make_generator_model()

    discriminator = make_discriminator_model()

    cond_gan = ConditionalGAN(generator, discriminator)
    logger.info("The cGAN model has been built correctly.")

    cond_gan.summary()
    cond_gan.plot_model()
    logger.info("The cGAN model has been plotted correctly.")

    #debug_cgan(cond_gan)

    logger.info("The work is done.")
    logger.handlers.clear()
