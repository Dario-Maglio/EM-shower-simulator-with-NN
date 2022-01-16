"""Subroutins for the creation of the generator and discriminator models."""

import os
import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (Input,
                                     Concatenate,
                                     Embedding,
                                     Dense,
                                     BatchNormalization,
                                     LeakyReLU,
                                     Reshape,
                                     Conv3DTranspose,
                                     MaxPooling3D,
                                     Conv3D,
                                     Dropout,
                                     Lambda,
                                     Concatenate,
                                     Multiply,
                                     Flatten)

#-------------------------------------------------------------------------------
"""Constant parameters of configuration and definition of global objects."""

# Configuration parameters
N_PID = 3
N_ENER = 30 + 1
NOISE_DIM = 1024
MBSTD_GROUP_SIZE = 8                                     #minibatch dimension
ENERGY_NORM = 6.503
ENERGY_SCALE = 1000000.
GEOMETRY = (12, 25, 25, 1)

# Define logger and handler
logger = logging.getLogger("ModelsLogger")

#-------------------------------------------------------------------------------
"""Subroutines for the generator network."""

def make_generator_model():
    """Define generator model:
    Input 1) Random noise from which the network creates a vector of images;
    Input 2) Energy label to be passed to the network;
    Input 3) ParticleID label to be passed to the network.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES * classes.
    """
    SIDE = 3
    N_FILTER = 32
    EMBED_DIM = 50
    KERNEL = (4, 5, 5)
    image_shape = (SIDE, SIDE, SIDE, N_FILTER)

    n_nodes = 1
    for cell in image_shape:
        n_nodes = n_nodes * cell

    # Input[i] -> input[i] + convolution * (KERNEL-1)
    error = "ERROR building the generator: shape different from geometry!"

    # Image generator input
    in_lat = Input(shape=(NOISE_DIM,), name="latent_input")
    li_lat = Reshape((1,NOISE_DIM))(in_lat)

    # Energy label input
    en_label = Input(shape=(1,), name="energy_input")
    li_en = Reshape((1, 1))(en_label)

    # Combine noise and energy
    gen = Multiply()([li_lat, li_en])
    gen = Dense(NOISE_DIM, activation="linear", use_bias=False)(gen)

    # ParticleID label input
    pid_label = Input(shape=(1,), name="particle_input")
    li_pid = Embedding(N_PID, EMBED_DIM)(pid_label)
    li_pid = Dense(EMBED_DIM, activation="relu", use_bias=False)(li_pid)

    # Merge image gen and label input  and at last particle ID
    merge = Concatenate()([gen, li_pid])
    gen = Dense(n_nodes, activation="relu", use_bias=False)(merge)
    gen = Reshape(image_shape)(gen)

    gen = Conv3DTranspose(2 * N_FILTER, KERNEL, strides=(1,2,2), use_bias=False)(gen)
    logger.info(gen.get_shape())
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(2 * N_FILTER, activation="linear", use_bias=False)(gen)

    gen = Conv3DTranspose(N_FILTER, KERNEL, strides=(1,2,2), use_bias=False)(gen)
    logger.info(gen.get_shape())
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(N_FILTER, activation="linear", use_bias=False)(gen)

    output = (Conv3DTranspose(1, KERNEL, use_bias=False, activation="tanh",
                                                        name="fake_image")(gen))

    logger.info(f"Shape of the generator output: {output.get_shape()}")
    assert output.get_shape().as_list()==[None, *GEOMETRY], error

    model = Model([in_lat, en_label, pid_label], output, name='generator')
    return model

def compute_energy(in_images):
    """Compute energy deposited in detector."""
    in_images = tf.cast(in_images, tf.float32)

    en_images = tf.math.multiply(in_images, ENERGY_NORM)
    en_images = tf.math.pow(10., en_images)
    en_images = tf.math.divide(en_images, ENERGY_SCALE)
    en_images = tf.math.reduce_sum(en_images, axis=[1,2,3])
    return en_images

def debug_generator(noise, verbose=False):
    """Uses the random seeds to generate fake samples and plots them."""
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info("Logging level set on DEBUG.")
    else:
        logger.setLevel(logging.WARNING)
        logger.info("Logging level set on WARNING.")
    logger.info("Start debugging the generator model.")

    generator = make_generator_model()
    data_images = generator(noise, training=False)
    logger.info(f"Shape of generated images: {data_images.shape}")

    energy = compute_energy(data_images)

    k=0
    plt.figure("Generated showers", figsize=(20,10))
    num_examples = data_images.shape[0]
    for i in range(num_examples):
       for j in range(data_images.shape[1]):
          k=k+1
          plt.subplot(num_examples, data_images.shape[1], k)
          plt.imshow(data_images[i,j,:,:,0]) #, cmap="gray")
          plt.axis("off")
    plt.show()

    for example in range(len(noise[0]) ):
      print(f"{example+1})\nPrimary particle = {int(noise[2][example][0])}"
           +f"\tInitial energy = {noise[1][example][0]}"
           +f"\tGenerated energy = {energy[example][0]}")

    logger.info("Debug of the generator model finished.")

#-------------------------------------------------------------------------------
"""Subroutines for the discriminator network."""

def make_discriminator_model():
    """Define discriminator model:
    Input 1) Vector of images associated to the given labels;
    Input 2) Energy label to be passed to the network;
    Input 3) ParticleID label to be passed to the network.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES * classes.
    """
    N_FILTER = 32
    KERNEL = (5, 7, 7)

    # padding="same" add a 0 to borders, "valid" use only available data !
    # Output of convolution = (input + 2padding - kernel) / strides + 1 !
    # Here we use padding default = "valid" (=0 above) and strides = 1 !
    # GEOMETRY[i] -> GEOMETRY[i] - 3convolution * (KERNEL[i] - 1) > 0 !
    error = "ERROR building the discriminator: smaller KERNEL is required!"

    n_nodes = 1
    for cell in GEOMETRY:
        n_nodes = n_nodes * cell

    # Image input
    in_image = Input(shape=GEOMETRY, name="input_image")

    discr = Conv3D(N_FILTER, KERNEL, use_bias=False)(in_image)
    logger.info(discr.get_shape())
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    discr = Conv3D(N_FILTER, KERNEL, use_bias=False)(discr)
    logger.info(discr.get_shape())
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    discr = Conv3D(2*N_FILTER, KERNEL, padding="same", use_bias=False)(discr)
    discr = MaxPooling3D(pool_size = KERNEL, padding ="same")(discr)
    logger.info(discr.get_shape())
    discr = Flatten()(discr)

    discr_conv = Dense(N_FILTER, activation="relu")(discr)
    discr_conv = Dense(N_FILTER, activation="relu")(discr_conv)
    output_conv = Dense(1, activation="sigmoid", name="decision")(discr_conv)

    discr_en = Dense(N_FILTER, activation="relu")(discr)
    discr_en = Dense(N_FILTER, activation="relu")(discr_en)
    output_en = Dense(1, activation="relu", name="energy_label")(discr_en)

    discr_id = Dense(N_FILTER, activation="relu")(discr)
    output_id = Dense(1, activation="sigmoid", name="particle_label")(discr_id)

    output = [output_conv, output_en, output_id]
    model = Model(in_image, output, name='discriminator')
    return model

def debug_discriminator(data, verbose=False):
    """Uses images from the sample to test discriminator model."""
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')
    logger.info("Start debugging discriminator model.")

    discriminator = make_discriminator_model()
    decision = discriminator(data)
    logger.info(f"\nDecision per raw:\n {decision[0]}")
    logger.info("Debug of the discriminator model finished.")
