""" Train the GAN with the Geant generated dataset and save the model """

# Examples of conditional GANs from which we built our neural network :
# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
# https://keras.io/examples/generative/conditional_gan/

import os
import time
import pathlib

from IPython import display # A command shell for interactive computing in Python.
from IPython.core.display import Image
from tensorflow.keras import Model
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
                                     Flatten)
from tensorflow.keras.utils import plot_model
#import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import uproot as up


BUFFER_SIZE = 1000
BATCH_SIZE = 100
NOISE_DIM = 1000
N_CLASSES_EN = 100
N_CLASSES_PID = 3
EMBED_DIM = 50

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

EPOCHS = 100
num_examples_to_generate = 5
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

generator_optimizer = tf.keras.optimizers.Adam(3e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(3e-4)

#-------------------------------------------------------------------------------

path = pathlib.Path("../../dataset/filtered_data/data_MVA.root").resolve()
file = up.open(path)#v"/content/EM-shower-simulator-with-NN/dataset/filtered_data/data_MVA.root"
branches = file["h"].arrays()

train_images = np.array(branches["shower"]).astype("float32")
#images in log10 scale, zeros like pixel are set to -5, maximum of pixel is 2.4E5 keV ≃ 5.4 in log scale
train_images = (train_images -0.2) / 5.2 # Normalize the images to [-1, 1]
train_images = np.reshape(train_images, (-1, 12, 12, 12, 1))

en_labels = np.array(branches["en_in"]).astype("float32")/2000000.0 #np.zeros(1000).astype("float32")#
en_labels = np.transpose(en_labels)
en_labels = np.reshape(en_labels, (1000,1))

pid_labels = np.array(branches["primary"])+1
pid_labels = np.transpose(pid_labels)
pid_labels = np.reshape(pid_labels, (1000,1))

# Batch and shuffle the data
train_dataset = ( tf.data.Dataset.from_tensor_slices((train_images, en_labels, pid_labels))
                .shuffle(BUFFER_SIZE).batch(BATCH_SIZE) )

def debug_shower():
    """
    Plot showers from the dataset
    """
    print(train_images.shape)
    #plt.figure(figsize=(20,150))
    k=0
    N_showers = 10
    for i in range(N_showers):
        for j in range(12):
            k=k+1
            plt.subplot(N_showers, 12, k)
            plt.imshow(train_images[i, j, :, :, 0], cmap="gray")
            plt.axis("off")
    plt.show()
#-------------------------------------------------------------------------------

def make_generator_model():
    """
    Define generator model:
    Input 1) Energy label to be passed to the network;
    Input 2) ParticleID label to be passed to the network;
    Input 3) Random noise from which the network creates a vector of images,
            associated to the given labels.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES_* classes.
    """
    # en label input
    en_label = Input(shape=(1,))
    # embedding for categorical input
    li_en = Embedding(N_CLASSES_EN, EMBED_DIM)(en_label)
    # linear multiplication
    n_nodes = 3 * 3 * 3
    li_en = Dense(n_nodes)(li_en)
    # reshape to additional channel
    li_en = Reshape((3, 3, 3, 1))(li_en)
    assert li_en.get_shape().as_list() == [None, 3, 3, 3, 1]

    # pid label input
    pid_label = Input(shape=(1,))
    # embedding for categorical input
    li_pid = Embedding(N_CLASSES_PID, EMBED_DIM)(pid_label)
    # linear multiplication
    n_nodes = 3 * 3 * 3
    li_pid = Dense(n_nodes)(li_pid)
    # reshape to additional channel
    li_pid = Reshape((3, 3, 3, 1))(li_pid)
    assert li_pid.get_shape().as_list() == [None, 3, 3, 3, 1]

    # image generator input
    in_lat = Input(shape=(NOISE_DIM,))
    # foundation for 12x12x12 image
    n_nodes = 256 * 3 * 3 * 3
    gen = Dense(n_nodes, use_bias=False)(in_lat)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((3, 3, 3, 256))(gen)
    assert gen.get_shape().as_list() == [None, 3, 3, 3, 256] # Note: None is the batch size

    # merge image gen and label input
    merge = Concatenate()([gen, li_en, li_pid])

    gen = Conv3DTranspose(128, (5, 5, 5), strides=(1, 1, 1), padding="same", use_bias=False)(merge)
    assert gen.get_shape().as_list() == [None, 3, 3, 3, 128]
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv3DTranspose(64, (5, 5, 5), strides=(2, 2, 2), padding="same", use_bias=False)(gen)
    assert gen.get_shape().as_list() == [None, 6, 6, 6, 64]
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    output = (Conv3DTranspose(1, (5, 5, 5), strides=(2, 2, 2), padding="same",
                              use_bias=False, activation="tanh")(gen))
    assert output.get_shape().as_list() == [None, 12, 12, 12, 1]

    model = Model([in_lat, en_label, pid_label], output)
    return model


def generator_loss(fake_output):
    """
    Definie generator loss: successes on fake samples (from generator)
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = make_generator_model()
plot_model(generator, to_file="generator.png", show_shapes=True)
#-------------------------------------------------------------------------------

def debug_generator():
    """
    Creates a random noise and labels and generate a sample
    """
    generator = make_generator_model()
    noise = tf.random.normal([1, NOISE_DIM])
    en_labels = np.random.rand(1)*99
    pid_labels = np.random.rand(1)*2
    generated_image = generator([noise, en_labels, pid_labels], training=False)
    print(generated_image.shape)

    plt.figure(figsize=(20,150))
    for j in range(12):
        plt.subplot(1, 12, j+1)
        plt.imshow(generated_image[0,j,:,:,0], cmap="gray")
        plt.axis("off")
    plt.show()

#-------------------------------------------------------------------------------

def make_discriminator_model():
    """
    Define discriminator model :
    Input 1) Energy label to be passed to the network;
    Input 2) ParticleID label to be passed to the network;
    Input 3) Vector of images associated to the given labels.

    Labels are given as scalars in input; then they are passed to an embedding
    layer that creates a sort of lookup-table (vector[EMBED_DIM] of floats) that
    categorizes the labels in N_CLASSES_* classes.
    """
    # en label input
    en_label = Input(shape=(1,))
    # embedding for categorical input
    li_en = Embedding(N_CLASSES_EN, EMBED_DIM)(en_label)
    # scale up to image dimensions with linear activation
    n_nodes = 12*12*12
    li_en = Dense(n_nodes)(li_en)
    # reshape to additional channel
    li_en = Reshape((12,12,12, 1))(li_en)

    # pid label input
    pid_label = Input(shape=(1,))
    # embedding for categorical input
    li_pid = Embedding(N_CLASSES_PID, EMBED_DIM)(pid_label)
    # scale up to image dimensions with linear activation
    n_nodes = 12*12*12
    li_pid = Dense(n_nodes)(li_pid)
    # reshape to additional channel
    li_pid = Reshape((12,12,12, 1))(li_pid)

    # image input
    in_image = Input(shape=(12,12,12,1))
    # concat label as a channel
    merge = Concatenate()([in_image, li_en, li_pid])
    # downsample

    discr = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding="same")(merge)
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    discr = Conv3D(64, (5, 5, 5), strides=(2, 2, 2), padding="same")(discr)
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    discr = Flatten()(discr)
    output = Dense(1, activation="sigmoid")(discr)

    model = Model([in_image, en_label, pid_label], output)
    return model

def discriminator_loss(real_output, fake_output):
    """
    Define discriminator loss : fails on fake samples(from generator)
    and successes on real samples.
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

discriminator = make_discriminator_model()
plot_model(discriminator, to_file="discriminator.png", show_shapes=True)
#-------------------------------------------------------------------------------

def debug_discriminator():
    """
    Checks discriminator output on a random sample generated by the generator
    """
    discriminator = make_discriminator_model()
    noise = tf.random.normal([1, NOISE_DIM])
    en_labels = np.random.rand(1)*99
    pid_labels = np.random.rand(1)*2
    generator = make_generator_model()
    generated_image = generator([noise,en_labels,pid_labels], training=False)
    decision = discriminator( [generated_image,en_labels,pid_labels] )
    print(decision)

#-------------------------------------------------------------------------------
# Create a folder where it saves rusults from training in form of chechpoints. Also,
# create a random seed, to be used during the evaluation of the cGAN

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_and_save_images(model, epoch, test_input):
    """
    Creates and saves images at each epoch. Arguments:
    -) model: model (cGAN) to be evaluated;
    -) epoch: epoch whose predictions are to be saved;
    -) test_input: constant input to the model in order to evaluate performances.
    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    # 1 - Generate images
    predictions = model(test_input, training=False)
    # 2 - Plot the generated images
    #plt.figure(figsize=(75,75))
    k=0
    for i in range(predictions.shape[0]):
      for j in range( predictions.shape[1]):
        k=k+1
        plt.subplot(num_examples_to_generate, predictions.shape[1], k)
        plt.imshow(predictions[i,j,:,:,0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.show()
    # 3 - Save the generated images
    plt.savefig(f"image_at_epoch_{epoch}.png")
    plt.show()

#-------------------------------------------------------------------------------

class ConditionalGAN(tf.keras.Model):
    """
    Class for a conditional GAN. It inherits keras.Model properties and functions.
    """
    def __init__(self, discriminator, generator, latent_dim):
        """
        Constructor.
        Inputs:
        discriminator = discriminator network;
        generator = generator network;
        latent_dim = dimensionality of the space from which the noise is randomly
                     produced.
        """
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.generator_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.discr_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        """
        Metrics of the network.
        """
        return [self.gen_loss_tracker, self.discr_loss_tracker]

    def compile(self):
        """
        Compile method for the network.
        """
        super(ConditionalGAN, self).compile()

    # tf.function annotation causes the function
    # to be "compiled" as part of the training

    def train_step(self, dataset):
        """
        Train step of the cGAN.
        Input: dataset = combined images vector and labels upon which the network
                         trained.
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
      """
      Define the training function of the cGAN.
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
      for epoch in range(epochs):
        print(f"EPOCH = {epoch}")
        start = time.time()
        for image_batch in dataset:
            self.train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(self.generator,
                                epoch + 1,
                                seed)

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print (f"Time for epoch {epoch + 1} is {time.time()-start} sec")

      display.clear_output(wait=True)
      generate_and_save_images(self.generator,
                             epochs,
                             seed)

#-------------------------------------------------------------------------------
if __name__=="__main__":
    #debug_shower()
    #debug_generator();
    #debug_discriminator();

    #cond_gan = ConditionalGAN(discriminator=discriminator, generator=generator,
    #                           latent_dim=NOISE_DIM)
    #cond_gan.compile()
    #cond_gan.fit(train_dataset, epochs=EPOCHS)
    #cond_gan.train(train_dataset, EPOCHS)

    """
    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

    # evaluate the model
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    """

    # save model and architecture to single file
    #cond_gan.save("model.h5")
    print("Model saved to disk")
