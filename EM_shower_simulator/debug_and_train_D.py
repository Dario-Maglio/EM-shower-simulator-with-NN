""" Train the GAN with the Geant generated dataset and save the model """

import os
import logging
from sys import exit

from dataset import data_path, data_pull, debug_data_pull, debug_shower
from make_models_D import num_examples, debug_generator, debug_discriminator
from make_models_D import make_generator_model, make_discriminator_model
from class_GAN_D import ConditionalGAN

# Creation of the default dataset path
# In the project folder
#DPATH = data_path
# In colab after cloning the repository
DPATH = os.path.join("EM-shower-simulator-with-NN", data_path)
# In this folder
#DPATH = os.path.join("..", data_path)

# Define logger and handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger = logging.getLogger("DEBUGLogger")
logger.addHandler(ch)

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
        train_data = debug_data_pull(path, num_examples=num_examples)
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

    cond_gan.train(train_dataset, epochs=100)
    #cond_gan.fit(train_dataset, epochs=100)

    cond_gan.create_generator()

    #file_name = "cGAN.h5"
    #save_path = "model_saves"
    #if not os.path.isdir(save_path):
    #   os.makedirs(save_path)
    #cond_gan.save(os.path.join(save_path, file_name))

    # evaluate the model???
    #scores = model.evaluate(X, Y, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


if __name__=="__main__":

    debug(verbose=True)

    train_cgan()

    logger.info("The work is done.")
    logger.handlers.clear()
