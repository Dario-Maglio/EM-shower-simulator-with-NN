""" Train the GAN with the Geant generated dataset and save the model """

# Examples of conditional GANs from which we built our neural network :
# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
# https://keras.io/examples/generative/conditional_gan/

import os
import time
import logging
from sys import exit
from pathlib import Path

import numpy as np
import uproot as up
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
                                     Flatten)

from IPython import display
from IPython.core.display import Image

#-------------------------------------------------------------------------------
"""Constant parameters of configuration and definition of global objects."""

VERBOSE = True

# Creation of the default dataset path
# In the project folder
data_path = os.path.join("dataset","filtered_data","data_MVA.root")
# In colab after cloning the repository
#DPATH = os.path.join("EM-shower-simulator-with-NN", data_path)
# In this folder
DPATH = os.path.join("..", data_path)

# Configuration of the dataset structure
MBSTD_GROUP_SIZE = 8                            #minibatch dimension
BATCH_SIZE = 256
BUFFER_SIZE = 10000
N_CLASSES_PID = 3
N_CLASSES_EN = 100 + 1
GEOMETRY = (12, 12, 12, 1)

# Configuration of the cGAN structure
EPOCHS = 1
L_RATE = 3e-4
NOISE_DIM = 1000
DEFAULT_D_OPTIM = tf.keras.optimizers.Adam(L_RATE)
DEFAULT_G_OPTIM = tf.keras.optimizers.Adam(L_RATE)

# Create a random seed, to be used during the evaluation of the cGAN.
#tf.random.set_seed(42)
num_examples_to_generate = 5                 #multiple or minor of minibatch
test_noise = [tf.random.normal([num_examples_to_generate, NOISE_DIM]),
              tf.random.uniform([num_examples_to_generate, 1], minval= 0.,
                                maxval=N_CLASSES_EN),
              tf.random.uniform([num_examples_to_generate, 1], minval= 0.,
                                maxval=N_CLASSES_PID)]

# Define logger and handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger = logging.getLogger("trainLogger")
logger.addHandler(ch)

#-------------------------------------------------------------------------------
"""Subroutines for the dataset organization and shower plot."""

def data_pull(path):
    """Organize and reshape the dataset for the cGan training.
    Take in input a path to the dataset and return an iterator over batchs that
    can be used to train the cGAN using the method train.
    """
    logger.info("Loading the training dataset.")
    try:
       with up.open(Path(path).resolve()) as file:
           branches = file["h"].arrays()
    except Exception as e:
       print("Error: Invalid path or corrupted file.")
       raise e

    train_images = np.array(branches["shower"]).astype("float32")
    # Images in log10 scale, zeros like pixel are set to -5,
    # maximum of pixel is 2.4E5 keV => maximum<7 in log scale
    N_EVENT = train_images.shape[0]
    # Normalize the images to [-1, 1] and reshape
    train_images = np.reshape((train_images - 1.)/6., (N_EVENT, *GEOMETRY))

    en_labels = np.array(branches["en_in"]).astype("float32")/1000000.0
    en_labels = np.transpose(en_labels)
    assert N_EVENT == en_labels.shape[0], "Dataset energy labels compromised!"
    en_labels = np.reshape(en_labels, (N_EVENT, 1))

    # Particle labels are -1, 0, 1 ==> 0, 1, 2 for embedding layer
    pid_labels = np.array(branches["primary"]) + 1
    pid_labels = np.transpose(pid_labels)
    assert N_EVENT == pid_labels.shape[0], "Dataset PID labels compromised!"
    pid_labels = np.reshape(pid_labels, (N_EVENT, 1))

    # Shuffle and batch the data
    train_dataset = (train_images, en_labels, pid_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
    return train_dataset

def debug_data_pull(path):
    """Import data images from the dataset and test shapes.
    Take in input a path to the dataset and return train_images from the dataset
    that can be plotted with debug_shower.
    """
    logger.info("Start debugging the dataset loading subroutines.")
    try:
       with up.open(Path(path).resolve()) as file:
           branches = file["h"].arrays()
    except Exception as e:
       print("Error: Invalid path or corrupted file.")
       raise e

    train_images = np.array(branches["shower"]).astype("float32")
    # Images in log10 scale, zeros like pixel are set to -5,
    # maximum of pixel is 2.4E5 keV => maximum<7 in log scale
    N_EVENT = train_images.shape[0]
    # Normalize the images to [-1, 1] and reshape
    train_images = np.reshape((train_images - 1.)/6., (N_EVENT, *GEOMETRY))

    en_labels = np.array(branches["en_in"]).astype("float32")
    en_labels = np.transpose(en_labels)
    assert N_EVENT == en_labels.shape[0], "Dataset energy labels compromised!"

    pid_labels = np.array(branches["primary"]) + 1
    pid_labels = np.transpose(pid_labels)
    assert N_EVENT == pid_labels.shape[0], "Dataset PID labels compromised!"
    logger.info("Debug of the loading subroutines finished.")
    return train_images

def debug_shower(data_images, num_examples=1):
    """Show num_examples showers from data_images and return the figure."""
    logger.info("Start debug data_images features.")
    logger.info(f"Shape of data_images: {data_images.shape}")
    k=0
    fig = plt.figure("Generated showers")
    for i in range(num_examples):
       for j in range(GEOMETRY[0]):
          k=k+1
          plt.subplot(num_examples, GEOMETRY[0], k)
          plt.imshow(data_images[i,j,:,:,0], cmap="gray")
          plt.axis("off")
    plt.show()
    logger.info("Debug of data_images features finished.")
    return fig

def generate_and_save_images(model, noise, epoch=0, verbose=False):
    """Create images generated by the model at each epoch from the input noise.
    If verbose=True it only shows the output images of the model in debug model.
    If verbose=False it also saves the output images in 'training_results/'.
    Arguments:
    -) model: model (cGAN) to be evaluated;
    -) noise: constant input to the model in order to evaluate performances;
    -) epoch: epoch whose predictions are to be saved.
    """
    # 1 - Generate images
    predictions = model(noise, training=False)
    logger.info(f"Shape of generated images: {predictions.shape}")


    # 2 - Plot the generated images
    fig = debug_shower(predictions, num_examples=predictions.shape[0])

    # 3 - Save the generated images
    if not verbose:
       save_path = Path('training_results').resolve()
       file_name = f"image_at_epoch_{epoch}.png"

       if not os.path.isdir(save_path):
          os.makedirs(save_path)

       fig.savefig(os.path.join(save_path, file_name))
    else:
       logger.info("Images not saved.")

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
    N_FILTER = 64
    EMBED_DIM = 50
    KERNEL = (4, 4, 4)
    input_shape = (3, 3, 3, 1)
    image_shape = (3, 3, 3, 3 * N_FILTER)

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

    gen = Conv3DTranspose(2*N_FILTER, KERNEL, use_bias=False)(merge)
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

def generator_loss(fake_output):
    """Definie generator loss:
    successes on fake samples from the generator valued as true samples by
    the discriminator fake_output.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def debug_generator(noise):
    """Uses the random seeds to generate fake samples and plots them using the
    generate_and_save_images subroutine.
    """
    logger.info("Start debugging the generator model.")
    generator = make_generator_model()
    generate_and_save_images(generator, noise, verbose=True)
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
        # Minibatch/batch must be divisible by (or smaller than) group_size.
        group_size = tf.minimum(group_size, tf.shape(discr)[0])
        # Input shape.
        shape = discr.shape
        # Split minibatch into M groups of size G.
        minib = tf.reshape(discr, [group_size, -1, shape[1], shape[2], shape[3], shape[4]])
        # Cast to FP32.
        minib = tf.cast(minib, tf.float32)
        # Subtract mean over group.
        minib = minib - tf.reduce_mean(minib, axis=0, keepdims=True)
        # Calculate variance over group.
        minib = tf.reduce_mean(tf.square(minib), axis=0)
        # Calculate std dev over group.
        minib = tf.sqrt(minib + 1e-8)
        # Take average over fmaps and pixels.
        minib = tf.reduce_mean(minib, axis=[1,2,3,4], keepdims=True)
        # Cast back to original data type.
        minib = tf.cast(minib, discr.dtype)
        # New tensor by replicating input multiples times.
        minib = tf.tile(minib, [group_size, shape[1], shape[2], shape[3], 1])
        # Append as new fmap.
        return tf.concat([discr, minib], axis=-1)

def make_discriminator_model():
    """Define discriminator model:
    Input 1) Vector of images associated to the given labels;
    Input 2) Energy label to be passed to the network;
    Input 3) ParticleID label to be passed to the network.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES_ * classes.
    """
    N_FILTER = 32
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

    discr = Conv3D(2 * N_FILTER - 1, KERNEL)(discr)
    logger.info(discr.get_shape())
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    minibatch = Lambda(minibatch_stddev_layer, name="minibatch")(discr)
    logger.info(f"Minibatch shape: {minibatch.get_shape()}")
    discr = Conv3D(4 * N_FILTER, KERNEL)(minibatch)
    logger.info(f"Shape of the last discriminator layer: {discr.get_shape()}")

    discr = Flatten()(discr)
    output = Dense(1, activation="sigmoid", name="decision")(discr)

    model = Model([in_image, en_label, pid_label], output, name='discriminator')
    return model

def discriminator_loss(real_output, fake_output):
    """Define discriminator loss:
    fails on fake samples (from generator) and successes on real samples.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def debug_discriminator(images):
    """Uses images from the sample to test discriminator model."""
    logger.info("Start debugging discriminator model.")
    discriminator = make_discriminator_model()
    images = images[0:num_examples_to_generate]
    logger.info(f"Shape of images: {images.shape}")
    en_label = tf.random.uniform([num_examples_to_generate, 1], minval= 0.,
                                maxval=N_CLASSES_EN),
    pid_label = tf.random.uniform([num_examples_to_generate, 1], minval= 0.,
                                maxval=N_CLASSES_PID)
    decision = discriminator([images, en_label, pid_label])
    logger.info(f"\nDecision per raw:\n {decision}")
    logger.info("Debug finished.")

#-------------------------------------------------------------------------------
"""Conditional GAN Class, attributes and methods."""

class ConditionalGAN(tf.keras.Model):
    """Class for a conditional GAN.
    It inherits keras.Model properties and functions.
    """
    def __init__(self, discrim, gener, discrim_optim, gener_optim):
        """Constructor.
        Inputs:
        discrim = discriminator network;
        gener = generator network;
        discrim_optim = discriminator optimizer;
        gener_optim = generator optimizer;
        """
        super(ConditionalGAN, self).__init__()
        self.discriminator = discrim
        self.generator = gener
        self.discriminator_optimizer = discrim_optim
        self.generator_optimizer = gener_optim
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.discr_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        """Metrics of the network."""
        return [self.gen_loss_tracker, self.discr_loss_tracker]

    def compile(self):
        """Compile method for the network."""
        super(ConditionalGAN, self).compile(optimizer = 'adam')

    def summary(self):
        """Summary method for both the generator and discriminator."""
        print("\n\n Conditional GAN model summary:\n")
        self.generator.summary()
        print("\n")
        self.discriminator.summary()

    # tf.function annotation causes the function
    # to be "compiled" as part of the training

    def train_step(self, dataset):
        """Train step of the cGAN.
        Input: dataset = combined images vector and labels upon which the network trained.
        Description:
        1 - Create a random noise to feed it into the model for the images generation ;
        2 - Generate images and calculate loss values using real images and labels ;
        3 - Calculate gradients using loss values and model variables;
        4 - Process Gradients and Run the Optimizer ;
        """
        real_images, real_en_labels, real_pid_labels = dataset
        #dummy_labels = real_labels[:, :, None, None, None]
        #dummy_labels = tf.repeat( dummy_labels, repeats=[12 * 12 * 12] )
        #dummy_labels = tf.reshape( dummy_labels, (-1, 12 ,12 , 12, N_CLASSES) )
        #print(dummy_labels.shape)
        random_noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        #noise_and_labels= tf.concat([random_noise, real_labels],axis=1)

        # GradientTape method records operations for automatic differentiation.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            generated_images = self.generator([random_noise, real_en_labels,
                                               real_pid_labels], training=True)

            #fake_image_and_labels = tf.concat([generated_images, dummy_labels], -1)
            #real_image_and_labels = tf.concat([real_images, dummy_labels], -1)

            real_output = self.discriminator([real_images, real_en_labels,
                                              real_pid_labels], training=True)
            fake_output = self.discriminator([generated_images, real_en_labels,
                                              real_pid_labels], training=True)

            gen_loss = generator_loss(fake_output)
            discr_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss,
                                          self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(discr_loss,
                                           self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                         self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                         self.discriminator.trainable_variables))

        # Monitor losses
        self.gen_loss_tracker.update_state(gen_loss)
        self.discr_loss_tracker.update_state(discr_loss)
        return{
            "gen_loss": self.gen_loss_tracker.result(),
            "discr_loss": self.discr_loss_tracker.result(),
        }

    def train(self, dataset, epochs):
        """Define the training function of the cGAN.
        Inputs:
        dataset = combined real images vectors and labels;
        epochs = number of epochs for the training.

        For each epoch:
        -) For each batch of the dataset, run the custom "train_step" function;
        -) Produce images;
        -) Save the model every 5 epochs as a checkpoint;
        -) Print out the completed epoch no. and the time spent;
        Then generate a final image after the training is completed.
        """

        # Create a folder to save rusults from training in form of checkpoints.
        checkpoint_dir = "./training_checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
                   generator=cond_gan.generator,
                   discriminator=cond_gan.discriminator,
                   generator_optimizer=cond_gan.generator_optimizer,
                   discriminator_optimizer=cond_gan.discriminator_optimizer)

        display.clear_output(wait=True)
        for epoch in range(epochs):
           print(f"Running EPOCH = {epoch + 1}")
           start = time.time()

           for image_batch in dataset:
              self.train_step(image_batch)

           display.clear_output(wait=True)
           print(f"EPOCH = {epoch + 1}")
           print (f"Time for epoch {epoch + 1} is {time.time()-start} sec")
           generate_and_save_images(self.generator, epoch + 1, test_noise)

           #if (epoch + 1) % 5 == 0:
           checkpoint.save(file_prefix = checkpoint_prefix)

#-------------------------------------------------------------------------------

def debug(path=DPATH, verbose=VERBOSE):

    if verbose :
        #Set the logger level to debug
        logger.setLevel(logging.DEBUG)
        logger.info('Logging level set on DEBUG.')
    else:
        logger.setLevel(logging.WARNING)
        logger.info('Logging level set on WARNING.')

    try:
        train_images = debug_data_pull(path)
    except AssertionError as e:
        print(f"An error occurred while loading the dataset: \n{e}")
        exit()
    except Exception as e:
        print(f"Error: Invalid path or corrupted file. \n{e}")
        exit()

    if verbose :
        #Execute debug subroutines
        #debug_shower(train_images, num_examples_to_generate)
        debug_generator(test_noise)
        debug_discriminator(train_images)

def train_cgan(path=DPATH, verbose=VERBOSE):

    logger.info("Starting training operations.")
    train_dataset = data_pull(path)
    generator = make_generator_model()
    plot_model(generator, to_file="cgan-generator.png", show_shapes=True)

    discriminator = make_discriminator_model()
    plot_model(discriminator, to_file="cgan-discriminator.png", show_shapes=True)

    cond_gan = ConditionalGAN(discriminator, generator, DEFAULT_D_OPTIM, DEFAULT_G_OPTIM)
    #cond_gan.summary()
    cond_gan.compile()
    logger.info("The cGAN model has been compiled correctly.")

    cond_gan.train(train_dataset, EPOCHS)

    # evaluate the model???
    #scores = model.evaluate(X, Y, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


if __name__=="__main__":

    debug()

    train_cgan()

    logger.info("The work is done.")
    logger.handlers.clear()
