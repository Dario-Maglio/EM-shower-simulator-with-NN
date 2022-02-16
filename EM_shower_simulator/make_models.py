""" Subroutins for the creation of the generator and discriminator models """

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
                                     Conv3D,
                                     MaxPooling3D,
                                     AveragePooling3D,
                                     Dropout,
                                     Lambda,
                                     Concatenate,
                                     Flatten)

#-------------------------------------------------------------------------------
"""Constant parameters of configuration and definition of global objects."""

# Configuration parameters
N_PID = 3
N_ENER = 30 + 1
NOISE_DIM = 1024
MBSTD_GROUP_SIZE = 32
ENERGY_NORM = 6.7404
ENERGY_SCALE = 1000000.
GEOMETRY = (12, 25, 25, 1)

# Define logger and handler
logMod = logging.getLogger("ModelsLogger")

#-------------------------------------------------------------------------------
"""General subroutines for the network."""

def compute_energy(in_images):
    """Compute energy deposited in detector."""
    in_images = tf.cast(in_images, tf.float32)
    en_images = tf.math.multiply(in_images, ENERGY_NORM)
    en_images = tf.math.pow(10., en_images)
    en_images = tf.math.divide(en_images, ENERGY_SCALE)
    en_images = tf.math.reduce_sum(en_images, axis=[1,2,3])
    return en_images

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
    BASE = 8
    FILTER = 32
    EMBED_DIM = 30
    KERNEL_L = (1, 8, 8)
    KERNEL_S = (3, 6, 6)
    n_nodes = BASE*BASE*BASE
    image_shape = (BASE, BASE, BASE, -1)

    # Input[i] -> input[i] + convolution * (KERNEL-1)
    error = "ERROR building the generator: shape different from geometry!"

    # Image generator input
    in_lat = Input(shape=(NOISE_DIM,), name="latent_input")
    li_lat = Reshape(image_shape)(in_lat)

    # Energy label input
    en_label = Input(shape=(1,), name="energy_input")
    li_en = Dense(n_nodes, activation="relu")(en_label)
    li_en = Dense(n_nodes, activation="relu")(li_en)
    li_en = Reshape(image_shape)(li_en)

    # ParticleID label input
    pid_label = Input(shape=(1,), name="particle_input")
    li_pid = Embedding(N_PID, EMBED_DIM)(pid_label)
    li_pid = Dense(n_nodes, activation="relu")(li_pid)
    li_pid = Reshape(image_shape)(li_pid)

    # Combine noise and particle ID
    gen = Concatenate()([li_lat, li_pid])
    logMod.info(gen.get_shape())

    gen = Conv3DTranspose(1, KERNEL_S, padding="same", activation="relu")(gen)
    logMod.info(gen.get_shape())

    # Combine image and energy
    gen = Concatenate()([gen, li_en])
    logMod.info(gen.get_shape())

    gen = Dense(FILTER, activation="relu")(gen)
    gen = Dense(2*FILTER, activation="relu")(gen)

    gen = Conv3DTranspose(2*FILTER, KERNEL_L)(gen)
    logMod.info(gen.get_shape())
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv3DTranspose(FILTER, KERNEL_S)(gen)
    logMod.info(gen.get_shape())
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    output = Conv3DTranspose(1, KERNEL_S, activation="tanh", name="image")(gen)

    logMod.info(f"Shape of the generator output: {output.get_shape()}")
    assert output.get_shape().as_list()==[None, *GEOMETRY], error

    model = Model([in_lat, en_label, pid_label], output, name='generator')
    return model

def debug_generator(noise, verbose=False):
    """Uses the random seeds to generate fake samples and plots them."""
    if verbose :
        logMod.setLevel(logging.DEBUG)
        logMod.info('Logging level set on DEBUG.')
    else:
        logMod.setLevel(logging.WARNING)
        logMod.info('Logging level set on WARNING.')
    logMod.info("Start debugging the generator model.")

    generator = make_generator_model()
    data_images = generator(noise, training=False)
    logMod.info(f"Shape of generated images: {data_images.shape}")

    energy = compute_energy(data_images)

    k=0
    plt.figure("Generated showers", figsize=(20,10))
    num_examples = data_images.shape[0]
    for i in range(num_examples):
        print(f"{i+1})\tPrimary particle={int(noise[2][i][0])}"
             +f"\tInitial energy ={noise[1][i][0]}"
             +f"\tGenerated energy ={energy[i]}\n")
        for j in range(data_images.shape[1]):
           k=k+1
           plt.subplot(num_examples, data_images.shape[1], k)
           plt.imshow(data_images[i,j,:,:,0])
           plt.axis("off")
    plt.show()

    logMod.info("Debug of the generator model finished.")

#-------------------------------------------------------------------------------
"""Subroutines for the discriminator network."""

def minibatch_stddev_layer(discr, group_size=MBSTD_GROUP_SIZE):
    """Minibatch discrimination layer is important to avoid mode collapse.
    Once it is wrapped with a Lambda Keras layer it returns an additional filter
    node with information about the statistical distribution of the group_size,
    allowing the discriminator to recognize when the generator strarts to
    replicate the same kind of event multiple times.

    Inspired by
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
    """
    with tf.compat.v1.variable_scope('MinibatchStddev'):
        # Input 0 dimension must be divisible by (or smaller than) group_size.
        group_size = tf.minimum(group_size, tf.shape(discr)[0])
        # Input shape.
        shape = discr.shape
        # Split minibatch into M groups of size G.
        minib = tf.reshape(discr, [group_size, -1, shape[1], shape[2], shape[3], shape[4]])
        # Cast to FP32.
        minib = tf.cast(minib, tf.float32)
        # Calculate the std deviation for each pixel over minibatch
        minib = tf.math.reduce_std(minib + 1E-6, axis=0)
        # Take average over fmaps and pixels.
        minib = tf.reduce_mean(minib, axis=[2,3,4], keepdims=True)
        # Cast back to original data type.
        minib = tf.cast(minib, discr.dtype)
        # New tensor by replicating input multiples times.
        minib = tf.tile(minib, [group_size, 1 , shape[2], shape[3], 1])
        # Append as new fmap.
        return tf.concat([discr, minib], axis=-1)

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
    KERNEL = (3, 6, 6)

    # padding="same" add a 0 to borders, "valid" use only available data !
    # Output of convolution = (input + 2padding - kernel) / strides + 1 !
    # Here we use padding default = "valid" (=0 above) and strides = 1 !
    # GEOMETRY[i] -> GEOMETRY[i] - n_convolution * (KERNEL[i] - 1) > 0 !
    error = "ERROR building the discriminator: smaller KERNEL is required!"

    # Image input
    in_image = Input(shape=GEOMETRY, name="input_image")

    discr = Conv3D(N_FILTER, KERNEL)(in_image)
    logMod.info(discr.get_shape())
    discr = LeakyReLU(alpha=0.2)(discr)
    discr = Dropout(0.3)(discr)

    discr = AveragePooling3D(pool_size=(2,2,2), padding="valid")(discr)

    minibatch = Lambda(minibatch_stddev_layer, name="minibatch")(discr)
    logMod.info(f"Minibatch shape: {discr.get_shape()}")

    discr = Conv3D(2*N_FILTER, KERNEL)(minibatch)
    logMod.info(discr.get_shape())
    discr = LeakyReLU(alpha=0.2)(discr)

    discr = MaxPooling3D(pool_size=(2,2,2), padding="valid")(discr)

    logMod.info(discr.get_shape())
    discr = Flatten()(discr)

    discr_conv = Dense(4*N_FILTER, activation="relu")(discr)
    discr_conv = Dense(4*N_FILTER, activation="relu")(discr_conv)
    output_conv = Dense(1, activation="sigmoid", name="decision")(discr_conv)

    discr_en = Dense(4*N_FILTER, activation="relu")(discr)
    discr_en = Dense(4*N_FILTER, activation="relu")(discr_en)
    output_en = Dense(1, activation="relu", name="energy_label")(discr_en)

    discr_id = Dense(4*N_FILTER, activation="relu")(discr)
    discr_id = Dense(4*N_FILTER, activation="sigmoid")(discr_id)
    output_id = Dense(1, activation="sigmoid", name="particle_label")(discr_id)

    output = [output_conv, output_en, output_id]
    model = Model(in_image, output, name='discriminator')
    return model

def debug_discriminator(data, verbose=False):
    """Uses images from the sample to test discriminator model."""
    if verbose :
        logMod.setLevel(logging.DEBUG)
        logMod.info('Logging level set on DEBUG.')
    else:
        logMod.setLevel(logging.WARNING)
        logMod.info('Logging level set on WARNING.')
    logMod.info("Start debugging discriminator model.")

    discriminator = make_discriminator_model()
    decision = discriminator(data)
    logMod.info(f"\nDecision per raw:\n {decision[0]}")
    logMod.info("Debug of the discriminator model finished.")
