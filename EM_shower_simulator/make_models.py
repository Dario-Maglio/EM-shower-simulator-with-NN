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
                                     Conv3D,
                                     Dropout,
                                     Lambda,
                                     Multiply,
                                     Flatten)

#-------------------------------------------------------------------------------
"""Constant parameters of configuration and definition of global objects."""

# Configuration of the models structure
NOISE_DIM = 1000
N_PID = 3
N_ENER = 30 + 1
GEOMETRY = (12, 12, 12, 1)
ENERGY_SCALE = 1000000.
ENERGY_NORM = 6.

# Create a random seed, to be used during the evaluation of the cGAN.
tf.random.set_seed(42)
num_examples = 6
test_noise = [tf.random.normal([num_examples, NOISE_DIM]),
              tf.random.uniform([num_examples, 1], minval= 0., maxval=N_ENER),
              tf.random.uniform([num_examples, 1], minval= 0., maxval=N_PID)]

# Define logger and handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger = logging.getLogger("ModelsLogger")
logger.addHandler(ch)

#-------------------------------------------------------------------------------
"""Subroutines for the generator network."""

def make_generator_model():
    """Define generator model:
    Input 1) Random noise from which the network creates a vector of images;
    Input 2) Energy label to be passed to the network;
    Input 3) ParticleID label to be passed to the network.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES_* classes.
    """
    N_FILTER = 2
    EMBED_DIM = 5
    KERNEL = (4, 4, 4)
    input_shape = (3, 3, 3, 2*N_FILTER)
    image_shape = (3, 3, 3, N_FILTER)

    # Input[i] -> input[i] + 3 convolution * (KERNEL-1) = GEOMETRY[i]!
    error = "ERROR building the generator: shape different from geometry!"
    assert KERNEL[0] == (GEOMETRY[0] - input_shape[0])/3 + 1, error
    assert KERNEL[1] == (GEOMETRY[1] - input_shape[1])/3 + 1, error
    assert KERNEL[2] == (GEOMETRY[2] - input_shape[2])/3 + 1, error

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

    gen = Conv3DTranspose(3*N_FILTER, KERNEL, use_bias=False)(merge)
    logger.info(gen.get_shape())
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv3DTranspose(N_FILTER, KERNEL, use_bias=False)(gen)
    logger.info(gen.get_shape())
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    output = (Conv3DTranspose(1, KERNEL, use_bias=False,
                              activation="tanh", name="Fake_image")(gen))

    logger.info(f"Shape of the generator output: {output.get_shape()}")
    assert output.get_shape().as_list()==[None, *GEOMETRY], error

    model = Model([in_lat, en_label, pid_label], output, name='generator')
    return model

def debug_generator(noise=test_noise, verbose=False):
    """Uses the random seeds to generate fake samples and plots them."""
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')
    logger.info("Start debugging the generator model.")

    generator = make_generator_model()
    data_images = generator(noise, training=False)
    logger.info(f"Shape of generated images: {data_images.shape}")
    energy = np.array(data_images)
    energy = (10.**(energy*ENERGY_NORM + 1.))/ENERGY_SCALE
    energy = np.sum(energy, axis=(1,2,3,4))

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

    if verbose:
        save_path = 'model_plot'
        if not os.path.isdir(save_path):
           os.makedirs(save_path)
        file_name = "debug_generator.png"
        path = os.path.join(save_path, file_name)
        plot_model(generator, to_file=path, show_shapes=True)
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
    categorizes the labels in N_CLASSES_ * classes.
    """
    N_FILTER = 2
    EMBED_DIM = 10
    KERNEL = (4, 4, 4)

    # padding="same" add a 0 to borders, "valid" use only available data !
    # Output of convolution = (input + 2padding - kernel) / strides + 1 !
    # Here we use padding default = "valid" (=0 above) and strides = 1 !
    # GEOMETRY[i] -> GEOMETRY[i] - 3convolution * (KERNEL[i] - 1) > 0 !
    error = "ERROR building the discriminator: smaller KERNEL is required!"
    assert KERNEL[0] < GEOMETRY[0]/3 + 1, error
    assert KERNEL[1] < GEOMETRY[1]/3 + 1, error
    assert KERNEL[2] < GEOMETRY[2]/3 + 1, error

    n_nodes = 1
    for cell in GEOMETRY:
        n_nodes = n_nodes * cell

    # Image input
    in_image = Input(shape=GEOMETRY, name="input_image")

    # En label input
    en_label = Input(shape=(1,), name="energy_input")
    li_en = Embedding(N_ENER, N_ENER*EMBED_DIM)(en_label)
    li_en = Dense(n_nodes)(li_en)
    li_en = Dense(n_nodes)(li_en)
    li_en = Reshape(GEOMETRY)(li_en)

    # Pid label input
    pid_label = Input(shape=(1,), name="particleID_input")
    li_pid = Embedding(N_PID, N_PID*EMBED_DIM)(pid_label)
    li_pid = Dense(n_nodes)(li_pid)
    li_pid = Dense(n_nodes)(li_pid)
    li_pid = Reshape(GEOMETRY)(li_pid)

    # Concat label as a channel
    merge = Concatenate()([in_image, li_en, li_pid])
    logger.info(merge.get_shape())

    discr = Conv3D(N_FILTER, KERNEL)(merge)
    logger.info(discr.get_shape())
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    discr = Conv3D(2 * N_FILTER, KERNEL)(discr)
    logger.info(discr.get_shape())
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    discr = Flatten()(discr)
    output = Dense(1, activation="sigmoid", name="decision")(discr)

    model = Model([in_image, en_label, pid_label], output, name='discriminator')
    return model

def debug_discriminator(data, verbose=False):
    """Uses images from the sample to test discriminator model."""
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')
    logger.info("Start debugging discriminator model.")

    save_path = 'model_plot'
    if not os.path.isdir(save_path):
       os.makedirs(save_path)

    discriminator = make_discriminator_model()
    decision = discriminator(data)
    logger.info(f"\nDecision per raw:\n {decision}")
    if verbose:
        file_name = "debug_discriminator.png"
        path = os.path.join(save_path, file_name)
        plot_model(discriminator, to_file=path, show_shapes=True)

    logger.info("Debug discriminator model finished.")
