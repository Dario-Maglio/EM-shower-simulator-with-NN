""" Routine to generate dataset from the trained GAN """

import os
import sys
import logging
import datetime
from array import array

import numpy as np
import tensorflow as tf
from ROOT import TFile, TTree

from make_models import make_generator_model, make_discriminator_model
from make_models import compute_energy
from class_GAN import ConditionalGAN

ch = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger = logging.getLogger("DEBUGLogger")
logger.addHandler(ch)

if __name__=="__main__":

    logger.setLevel(logging.DEBUG)

    num_examples = 2000

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    cond_gan = ConditionalGAN(generator, discriminator)
    logger.info("The cGAN model has been built correctly.")

    generator, discriminator = cond_gan.restore()
    logger.info("Results restored correctly")

    noise = cond_gan.generate_noise(num_examples)

    logger.info("Beginning generating fake showers")
    start = datetime.datetime.now()
    predictions = generator(noise, training=False)
    elapsed = (datetime.datetime.now() - start)
    print(f"Generated {num_examples} showers in\
     {elapsed.seconds + elapsed.microseconds/1E6} s")

    energies = compute_energy(predictions)

    gan_dataset = os.path.join("..","dataset","gan_data","data_GAN_parte5.root")

    shower_in = array("d", 12*25*25*[0])
    primary_id_in = array("i", [0])
    primary_en_in = array("d", [0])
    deposit_en_in = array("d", [0])
    evt_in = array("i",[0])

    file = TFile(gan_dataset, "recreate")
    tree = TTree("h","ttree")
    tree.Branch("evt", evt_in,"evt/I")
    tree.Branch("primary",primary_id_in, "primary/I")
    tree.Branch("en_in",  primary_en_in, "en_in/D")
    tree.Branch("en_mis", deposit_en_in, "en_mis/D")
    tree.Branch("shower", shower_in, "shower[12][25][25][1]/D")

    shower = predictions.numpy() #awkward0.fromiter(
    primary_id = np.cast[np.int32](noise[2].numpy()) - 1
    primary_en = noise[1].numpy()
    deposit_en = energies.numpy()
    # print(primary_id)#shower,primary_en, deposit_en
    # print(predictions.shape)

    for i in range(num_examples):
        j=0
        evt_in[0] = i
        primary_en_in[0] = primary_en[i]*1E6
        primary_id_in[0] = int(primary_id[i])
        # print(primary_id_in[0])
        deposit_en_in[0] = deposit_en[i]*1E6
        for layer in range(12):
            for num_z in range(25):
                for num_y in range(25):
                    # print(j)
                    shower_in[j] = shower[i,layer,num_z,num_y,0]
                    j = j+1
        tree.Fill()

    tree.Write()
    file.Close()
