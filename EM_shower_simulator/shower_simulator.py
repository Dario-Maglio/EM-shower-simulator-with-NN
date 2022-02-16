""" Simulation of EM shower with a pre-trained NN """

import time
import logging
import argparse

from numpy import array, random
import tensorflow as tf

from .class_GAN import ConditionalGAN
from .make_models import make_generator_model, make_discriminator_model

DESCRIPTION = """
This command allows the user to generate a shower simulation from command line.
Furthermore, it allows to pass the shower's features as float arguments in the
order: energy particleID
It gives an error if a different input size or type is passed."""
DESCRIPTION_F = """insert the shower's features from command line"""

n_features = 2
default_features = random.rand(n_features)

def simulate_shower(features=default_features, verbose=0):
    """Given the input features as a list of n_features float components,
       it generates the corresponding shower simulation.
    """
    # Check the input format
    if not(len(features)==n_features):
        error = f'Expected input dimension {n_features}, not {len(features)}.'
        raise TypeError(error)

    print(f'simulating event with features: {features}')

    # Define logger and handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger = logging.getLogger("simulationLogger")
    logger.addHandler(ch)

    if verbose==1:
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    # Load the model
    try:
       logger.info('Loading the model...')
       gen = make_generator_model()
       dis = make_discriminator_model()
       gan = ConditionalGAN(gen, dis)
       logger.info('GAN created...')
       gan.evaluate()
    except Exception as error:
       logger.info(f'Missing model: {error}')
       return 1

    # Start simulation
    start_time = time.time()
    noise = gan.generate_noise(1)
    noise[1] = tf.constant(features[0], shape=(1,1))
    noise[2] = tf.constant(features[1], shape=(1,1))
    gan.generate_and_save_images(noise)
    time_elapsed = time.time() - start_time
    logger.info(f'Done in {time_elapsed:.4} seconds')
    logger.handlers.clear()
    return 0


def cli():
    """Command line interface (CLI) of the shower simulator."""
    PARSER = argparse.ArgumentParser(
       prog='simulate-EM-shower',
       formatter_class=argparse.RawTextHelpFormatter,
       description='Simulate EM shower\n' + DESCRIPTION,
       epilog="Further information in package's documentation")
    PARSER.add_argument('-v', '--verbose', action='store_true', help='DEBUG')
    PARSER.add_argument('-f', '--features', metavar='feature', action='store',
       nargs='+', default=default_features, help=DESCRIPTION_F)
    ARGS = PARSER.parse_args()

    # Arguments conversion to float
    inputs = []
    for item in ARGS.features:
        try:
           inputs.append(float(item))
        except:
           print(f'Unexpected value {item}.')

    STATE = simulate_shower(inputs, verbose=ARGS.verbose)
    if STATE == 1:
        print('Error while loading the model.')
    else:
        print('The work is done.')


if __name__ == "__main__":
    cli()
