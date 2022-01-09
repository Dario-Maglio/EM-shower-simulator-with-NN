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
MBSTD_GROUP_SIZE = 32                                     #minibatch dimension
NOISE_DIM = 1000
N_CLASSES_PID = 3
N_CLASSES_EN = 30 + 1
GEOMETRY = (12, 12, 12, 1)
ENERGY_SCALE = 1000000
ENERGY_NORM = 6.

# Create a random seed, to be used during the evaluation of the cGAN.
tf.random.set_seed(42)
num_examples = 6                 #multiple or minor of minibatch
test_noise = [tf.random.normal([num_examples, NOISE_DIM]),
              tf.random.uniform([num_examples, 1], minval= 0.,
                                maxval=N_CLASSES_EN),
              tf.random.uniform([num_examples, 1], minval= 0.,
                                maxval=N_CLASSES_PID)]

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
    N_FILTER = 16
    EMBED_DIM = 50
    KERNEL = (4, 4, 4)
    input_shape = (3, 3, 3, N_FILTER)
    image_shape = (3, 3, 3, 4 * N_FILTER)

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
    li_en = Embedding(N_CLASSES_EN, EMBED_DIM)(en_label)
    li_en = Dense(n_nodes)(li_en)
    li_en = Reshape(input_shape)(li_en)

    # ParticleID label input
    pid_label = Input(shape=(1,), name="particleID_input")
    li_pid = Embedding(N_CLASSES_PID, EMBED_DIM)(pid_label)
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

    if verbose :
        save_path = 'model_plot'
        if not os.path.isdir(save_path):
           os.makedirs(save_path)
        file_name = "debug_generator.png"
        path = os.path.join(save_path, file_name)
        plot_model(generator, to_file=path, show_shapes=True)
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
        minib = tf.math.reduce_std(minib, axis=0)
        # Take average over fmaps and pixels.
        minib = tf.reduce_mean(minib, axis=[1,2,3], keepdims=True)
        # Cast back to original data type.
        minib = tf.cast(minib, discr.dtype)
        # New tensor by replicating input multiples times.
        minib = tf.tile(minib, [group_size, shape[1], shape[2], shape[3], 1])
        # Append as new fmap.
        return tf.concat([discr, minib], axis=-1)

def auxiliary_condition(layer):
    """Auxiliary condition for energy deposition:
    Calculate the energy deposited in the detector en_image and pass to the
    discriminator the conditioned probability output_pdf for such an event given
    the initial energy label. The distribition is assumed to be a Gaussian
    centered in en_label with a std deviation of en_label/10, scalata per 0.2 e
    sommata a 0.8. Fatto per rendere possibile il train del generatore: se fosse
    solo la gaussiana, il discriminatore sarebbe troppo intelligente e userebbe
    solo questa informazione per disciminare esempi veri da falsi.
    """

    en_label = layer[0]
    en_label = tf.cast(en_label, tf.float32)
    in_image = layer[1]
    in_image = tf.cast(in_image, tf.float32)

    en_image = tf.math.multiply(in_image, ENERGY_NORM)
    en_image = tf.math.add(en_image, 1.)
    en_image = tf.math.pow(10., en_image)
    en_image = tf.math.divide(en_image, ENERGY_SCALE)
    en_image = tf.math.reduce_sum(en_image, axis=[1,2,3])

    output_pdf = tf.math.pow((en_label-en_image)/(en_label*0.1) , 2)
    output_pdf = tf.math.multiply(output_pdf, -0.5)
    output_pdf = tf.math.exp(output_pdf)
    output_pdf = tf.math.multiply(output_pdf, 0.2)
    output_pdf = tf.math.add(output_pdf, 0.8)
    return output_pdf

def make_discriminator_model():
    """Define discriminator model:
    Input 1) Vector of images associated to the given labels;
    Input 2) Energy label to be passed to the network;
    Input 3) ParticleID label to be passed to the network.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES_ * classes.
    """
    N_FILTER = 16
    EMBED_DIM = 50
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
    li_en = Embedding(N_CLASSES_EN, EMBED_DIM)(en_label)
    li_en = Dense(n_nodes)(li_en)
    li_en = Reshape(GEOMETRY)(li_en)

    # Pid label input
    pid_label = Input(shape=(1,), name="particleID_input")
    li_pid = Embedding(N_CLASSES_PID, EMBED_DIM)(pid_label)
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

    minibatch = Lambda(minibatch_stddev_layer, name="minibatch")(discr)
    logger.info(f"Minibatch shape: {minibatch.get_shape()}")
    discr = Conv3D(8 * N_FILTER, KERNEL)(minibatch)
    logger.info(f"Shape of the last discriminator layer: {discr.get_shape()}")

    discr = Flatten()(discr)
    discr = Dense(1, activation="sigmoid", name="decision")(discr)

    aux_output = Lambda(auxiliary_condition, name= "aux_condition")([en_label, in_image])

    output = Lambda(lambda x : x[0]*x[1], name= "final_decision")((aux_output, discr))

    model = Model([in_image, en_label, pid_label], output, name='discriminator')
    return model

def make_discriminator_model_2(verbose=False):
    """Define discriminator model:
    Input 1) Vector of images associated to the given labels;
    Input 2) Energy label to be passed to the network;
    Input 3) ParticleID label to be passed to the network.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES_ * classes.
    """
    N_FILTER = 32
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
    li_en = Dense(n_nodes)(en_label)
    li_en = Reshape(GEOMETRY)(li_en)

    # Pid label input
    pid_label = Input(shape=(1,), name="particleID_input")
    li_pid = Embedding(N_CLASSES_PID, EMBED_DIM * N_CLASSES_PID)(pid_label)
    li_pid = Dense(n_nodes)(li_pid)
    li_pid = Reshape(GEOMETRY)(li_pid)

    # Concat label as a channel
    merge = Concatenate()([in_image, li_en, li_pid])
    logger.info(merge.get_shape())

    # Discr main channel
    discr = Conv3D(N_FILTER, KERNEL)(merge)
    logger.info(discr.get_shape())
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    discr = Conv3D(2 * N_FILTER, KERNEL)(discr)
    logger.info(f"Shape of the last conv-discr layer: {discr.get_shape()}")
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    # Minibatch channel
    minibatch = Lambda(minibatch_stddev_layer, name="minibatch")(discr)
    logger.info(f"Minibatch shape: {minibatch.get_shape()}")
    minibatch = Conv3D(N_FILTER, KERNEL)(minibatch)
    logger.info(f"Shape of the conv-minibatch layer: {minibatch.get_shape()}")

    # Minibatch output
    minibatch = Flatten()(minibatch)
    minibatch = Dense(1, activation="sigmoid", name="repetitions")(minibatch)

    # Discriminator output
    discr = Flatten()(discr)
    discr = Dense(1, activation="sigmoid", name="decision")(discr)

    # Auxiliary output
    aux_output = Lambda(auxiliary_condition, name= "auxiliary")([en_label, in_image])

    # Output
    merge = Concatenate()([discr, minibatch, aux_output])
    out1 = Dense(1)(merge)
    out2 = Dense(1)(merge)
    mix = Multiply()([out1, out2])
    output = Dense(1, activation="sigmoid", name= "final_decision")(mix)

    model = Model([in_image, en_label, pid_label], output, name='discriminator')
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

    save_path = 'model_plot'
    if not os.path.isdir(save_path):
       os.makedirs(save_path)

    discriminator = make_discriminator_model()
    decision = discriminator(data)
    logger.info(f"\nDecision per raw:\n {decision}")
    if verbose :
        file_name = "debug_discriminator.png"
        path = os.path.join(save_path, file_name)
        plot_model(discriminator, to_file=path, show_shapes=True)

    discriminator = make_discriminator_model_2()
    decision = discriminator(data)
    logger.info(f"\nDecision 2 per raw:\n {decision}")
    if verbose :
        file_name = "debug_discriminator_2.png"
        path = os.path.join(save_path, file_name)
        plot_model(discriminator, to_file=path, show_shapes=True)
    logger.info("Debug discriminator model finished.")
