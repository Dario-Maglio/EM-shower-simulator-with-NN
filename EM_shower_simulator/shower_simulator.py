"""Simulation of EM shower with a pre-trained NN"""

import time
import logging

from numpy import array, reshape
from tensorflow.keras.models import load_model

n_features = 3
default_features = [21.2, 34.6, 12.]

def simulate_shower(features=default_features, verbose=0):
    """Given the input features as a list of n_features float components,
       it generates the corresponding shower simulation."""

    #Check the input format
    features = array(features)
    if not(features.dtype=='float64'):
        raise TypeError('Expected array of float as input.')
    if not(len(features)==n_features):
        e = f'Expected input dimension {n_features}, not {len(features)}.'
        raise TypeError(e)

    print(f'simulating event with features {features}')

    #Define logger and handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    #Define the logger
    logger = logging.getLogger("simulationLogger")
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
    simulate_shower()
