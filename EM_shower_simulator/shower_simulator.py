"""Simulation of EM shower with a pre-trained NN"""

import time
import logging
import argparse

from numpy import array, reshape
from tensorflow.keras.models import load_model

PACKAGE_NAME = 'EM_shower_simulator'
AUTHOR = 'Dario Cafasso, Daniele Passaro'
DESCRIPTION = '....description......'
URL = '....url.....'
TAG = '1.0.0'
n_features = 3
DEFAULT_FEATURES = [21.2, 34.6, 12.]

#Problemi: vorrei importare __version__(TAG) e costanti direttamente da init e config.

def simulate_shower(features, verbose=0):
    """Given the input features as a list of n_features float components,
       it generates the corresponding shower simulation."""

    #Check the input format
    features = array(features)
    if not(features.dtype=='float64'):
        raise TypeError('Expected array of float as input.')
    if not(len(features)==n_features):
        e = f'Expected input dimension {n_features}, not {len(features)}.'
        raise TypeError(e)

    #Define logger and handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    #Define the logger
    logger = logging.getLogger("LocalLog")
    logger.addHandler(ch)
    #Set the logger level to debug
    if verbose==1:
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    #Start operations
    logger.info('Loading the model...')
    start_time = time.time()
    try:
       #model = load_model('EM_shower_network/generator.h5')
       #model.predict(features)
       logger.info('Missing model')
    except:
       return 1
    time_elapsed = time.time()- start_time
    logger.info(f'Done in {time_elapsed:.4} seconds')
    logger.handlers.clear()
    #show results
    return 0


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
       prog=PACKAGE_NAME,
       formatter_class=argparse.RawTextHelpFormatter,
       description=PACKAGE_NAME + '\n' + DESCRIPTION,
       epilog=f'Further information: {URL}')
    PARSER.add_argument('-v', '--version', action='version', version=TAG)
    PARSER.add_argument('-d', '--verbose', action='store_true', help='DEBUG')
    PARSER.add_argument('-f', '--features', metavar='feature', action='store',
       nargs='+', default=DEFAULT_FEATURES,
       help="allow the user to insert the shower's features from command line"
           +"\nthe argument must be a list containing the features in the "
           +"following order:\n energy momentum angle")
    ARGS = PARSER.parse_args()

    #Arguments conversion to float
    inputs = []
    for item in ARGS.features:
        try:
           inputs.append(float(item))
        except:
           print(f'Unexpected value {item}.')

    #Start simulation
    STATE = simulate_shower(inputs, verbose=ARGS.verbose)
    if STATE == 1:
        print('Error while loading the model.')
    else:
        print('The work is done.')
