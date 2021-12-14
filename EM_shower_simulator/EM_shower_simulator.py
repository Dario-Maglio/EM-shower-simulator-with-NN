"""Simulation of EM shower with a pre-trained NN"""

from .config import *
from .__version__ import *

import time
import logging
import argparse

from numpy import array


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

    #Define the logger handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    #Define the logger
    logger = logging.getLogger("LocalLog")
    logger.addHandler(ch)
    #Set the logger level to debug
    if verbose==1:
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')

    #Start operations
    logger.info('Loading model and weights...')
    start_time = time.time()
    #Things happen------------------------------------here
    time_elapsed = time.time()- start_time
    logger.info(f'Done in {time_elapsed:.3} seconds')
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
           +"following order: energy momentum angle")
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
        print('A problem occurred in loading weights')
    elif STATE == 2:
        print('A problem occurred')
    else:
        print('The work is done.')
