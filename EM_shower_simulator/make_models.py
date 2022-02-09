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
NOISE_DIM = 512
MBSTD_GROUP_SIZE = 32                                     #minibatch dimension
ENERGY_NORM = 6.7404
ENERGY_SCALE = 1000000.
GEOMETRY = (12, 25, 25, 1)
# EN_CUTOFF = 1. # keV

# Define logger and handler
logger = logging.getLogger("ModelsLogger")

#-------------------------------------------------------------------------------
"""Subroutines for the generator network."""
def zero_suppression(output):
    """Zero suppression on output images
    """
    cutoff = tf.zeros_like(output)
    void_pixel = -tf.ones_like(output)

    def f_true(): return output
    def f_false(): return void_pixel

    output = tf.where( tf.greater(output , cutoff), output, void_pixel )
    return output

def make_generator_model():
    """Define generator model:
    Input 1) Random noise from which the network creates a vector of images;
    Input 2) Energy label to be passed to the network;
    Input 3) ParticleID label to be passed to the network.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES_* classes.
    """
    N_FILTER = 16
    EMBED_DIM = 10
    KERNEL = (4, 7, 7)
    input_shape = (3, 7, 7, N_FILTER)
    image_shape = (3, 7, 7, 4*N_FILTER)

    # Input[i] -> input[i] + 3 convolution * (KERNEL-1) = GEOMETRY[i]!
    error = "ERROR building the generator: shape different from geometry!"
    #assert KERNEL[0] == (GEOMETRY[0] - input_shape[0])/3 + 1, error
    #assert KERNEL[1] == (GEOMETRY[1] - input_shape[1])/3 + 1, error
    #assert KERNEL[2] == (GEOMETRY[2] - input_shape[2])/3 + 1, error

    n_nodes = 1
    for cell in input_shape:
        n_nodes = n_nodes * cell

    # Energy label input
    en_label = Input(shape=(1,), name="energy_input")
    li_en = Embedding(N_ENER, N_ENER*EMBED_DIM)(en_label)
    li_en = Dense(n_nodes)(li_en)
    li_en = Reshape(input_shape)(li_en)

    # ParticleID label input
    pid_label = Input(shape=(1,), name="particleID_input")
    li_pid = Embedding(N_PID, N_PID*EMBED_DIM)(pid_label)
    li_pid = Dense(n_nodes)(li_pid)
    li_pid = Reshape(input_shape)(li_pid)

    n_nodes = 1
    for cell in image_shape:
        n_nodes = n_nodes * cell

    # Image generator input
    in_lat = Input(shape=(NOISE_DIM,), name="latent_input")
    gen = Dense(n_nodes, use_bias=False)(in_lat)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape(image_shape)(gen)

    # Merge image gen and label input
    merge = Concatenate()([gen, li_en, li_pid])

    gen = Conv3DTranspose(6*N_FILTER, KERNEL, use_bias=False)(merge)
    logger.info(gen.get_shape())
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv3DTranspose(2*N_FILTER, KERNEL, use_bias=False)(gen)
    logger.info(gen.get_shape())
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    output = (Conv3DTranspose(1, KERNEL, use_bias=False,
                              activation="tanh", name="Fake_image")(gen))
    output = Lambda(zero_suppression, name="Fake_image_zero_suppression")(output)
    logger.info(f"Shape of the generator output: {output.get_shape()}")
    assert output.get_shape().as_list()==[None, *GEOMETRY], error

    # print(output)

    model = Model([in_lat, en_label, pid_label], output, name='generator')
    return model

def debug_generator(noise, verbose=False):
    """Uses the random seeds to generate fake samples and plots them."""
    if verbose :
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')
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
      print(f"{example+1})\tPrimary particle={int(noise[2][example][0])}"
           +f"\tInitial energy ={noise[1][example][0]}"
           +f"\tGenerated energy ={energy[example]}")

    logger.info("Debug of the generator model finished.")

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
        # print(f"STD DEVIATION \n{minib}")
        # Take average over fmaps and pixels.
        minib = tf.reduce_mean(minib, axis=[2,3,4], keepdims=True)
        # print(f"MEAN \n{minib}")
        # Cast back to original data type.
        minib = tf.cast(minib, discr.dtype)
        # New tensor by replicating input multiples times.
        minib = tf.tile(minib, [group_size, 1 , shape[2], shape[3], 1])
        #print(f"SHAPE MINIBATCH {minib.shape}")
        # Append as new fmap. # which axis ????
        return tf.concat([discr, minib], axis=-1)

def compute_energy(in_images):
    """Compute energy deposited in detector
    """
    in_images = tf.cast(in_images, tf.float32)

    en_images = tf.math.multiply(in_images, ENERGY_NORM)
    en_images = tf.math.pow(10., en_images)
    en_images = tf.math.divide(en_images, ENERGY_SCALE)
    en_images = tf.math.reduce_sum(en_images, axis=[1,2,3])
    return en_images

def energies_per_layer(in_images):
    """Compute energy deposited in detector for each layer
    """
    in_images = tf.cast(in_images, tf.float32)
    shape = in_images.shape

    en_images = tf.math.multiply(in_images, ENERGY_NORM)
    en_images = tf.math.pow(10., en_images)
    en_images = tf.math.divide(en_images, ENERGY_SCALE)
    en_images = tf.math.reduce_sum(en_images, axis=[2,3,4])#, keepdims=True)

    #en_images = tf.tile(en_images, [1, 1 , shape[2], shape[3], 1])
    #output: (None, 12)
    return  en_images#tf.concat([in_images, en_images], axis=-1)


def make_discriminator_model():
    """Define discriminator model:
    Input 1) Vector of images associated to the given labels;
    Input 2) Energy label to be passed to the network;
    Input 3) ParticleID label to be passed to the network.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES * classes.
    """
    N_FILTER = 64
    KERNEL = (5, 5, 5)
    KERNEL_1 = (3, 3, 3)
    KERNEL_2 = (2, 2, 2)
    # padding="same" add a 0 to borders, "valid" use only available data !
    # Output of convolution = (input + 2padding - kernel) / strides + 1 !
    # Here we use padding default = "valid" (=0 above) and strides = 1 !
    # GEOMETRY[i] -> GEOMETRY[i] - 3convolution * (KERNEL[i] - 1) > 0 !
    error = "ERROR building the discriminator: smaller KERNEL is required!"
    # assert KERNEL[0] < GEOMETRY[0]/3 + 1, error
    # assert KERNEL[1] < GEOMETRY[1]/3 + 1, error
    # assert KERNEL[2] < GEOMETRY[2]/3 + 1, error

    n_nodes = 1
    for cell in GEOMETRY:
        n_nodes = n_nodes * cell

    # Image input
    in_image = Input(shape=GEOMETRY, name="input_image")

    energies = Lambda(energies_per_layer, name="energies_per_layer")(in_image)
    energies = Dense(2*N_FILTER, activation="relu")(energies)

    # minibatch = Lambda(minibatch_stddev_layer, name="minibatch")(in_image)
    # logger.info(f"Minibatch shape: {discr.get_shape()}")

    discr = Conv3D(N_FILTER, KERNEL, use_bias=False)(in_image)#in_image
    logger.info(discr.get_shape())
    discr = AveragePooling3D(pool_size = (2,2,2), padding ="valid")(discr)
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    minibatch = Lambda(minibatch_stddev_layer, name="minibatch")(discr)
    logger.info(f"Minibatch shape: {discr.get_shape()}")

    discr = Conv3D(2*N_FILTER, (2,2,2) , padding="valid", use_bias=False)(minibatch)
    discr = MaxPooling3D(pool_size = (2,2,2) , padding ="valid")(discr)
    logger.info(discr.get_shape())
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    # discr = Conv3D(3*N_FILTER, (2,2,2), padding="same", use_bias=False)(discr)
    # discr = MaxPooling3D(pool_size = (2,2,2) , padding ="same")(discr)
    # discr = LeakyReLU()(discr)
    # discr = Dropout(0.3)(discr)

    # minibatch = Lambda(minibatch_stddev_layer, name="minibatch")(discr)
    # logger.info(f"Minibatch shape: {minibatch.get_shape()}")

    logger.info(discr.get_shape())
    discr = Flatten()(discr)
    discr = Concatenate()([discr, energies])

    discr_conv = Dense(N_FILTER, activation="relu")(discr)
    discr_conv = Dense(N_FILTER, activation="relu")(discr_conv)
    output_conv = Dense(1, activation="sigmoid", name="decision")(discr_conv)

    discr_en = Dense(N_FILTER, activation="relu")(discr)
    discr_en = Dense(N_FILTER, activation="relu")(discr_en)
    output_en = Dense(1, activation="relu", name="energy_label")(discr_en)

    #total_energy = Lambda(compute_energy, name="total_energy")(in_image)

    discr_id = Dense(N_FILTER, activation="relu")(discr)
    discr_id = Dense(N_FILTER, activation="sigmoid")(discr_id)
    output_id = Dense(1, activation="sigmoid", name="one_hot")(discr_id)

    model = Model(in_image, [output_conv, output_en, output_id], name='discriminator')#, pid_label
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
