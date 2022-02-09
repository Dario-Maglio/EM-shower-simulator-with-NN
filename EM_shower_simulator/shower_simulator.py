""" Simulation of EM shower with a pre-trained NN """

import time
import logging
import argparse

from numpy import array, random

DESCRIPTION = """
This command allows the user to generate a shower simulation from command line.
Furthermore, it allows to pass the shower's features as float arguments in the
order: energy momentum angle
It gives an error if a different input size or type is passed."""
DESCRIPTION_F = """insert the shower's features from command line"""

n_features = 2
default_features = random.rand(n_features)

def simulate_shower(features=default_features, verbose=0):
    """Given the input features as a list of n_features float components,
       it generates the corresponding shower simulation.
    """
    # Check the input format
    features = array(features)
    if not(features.dtype=='float64'):
        raise TypeError('Expected array of float as input.')
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
       #model = load_model(os.path.join("model_save","cGAN.h5"))
    except:
       logger.info('Missing model')
       return 1

    # Start simulation
    start_time = time.time()
    #model.simulate(features)
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
