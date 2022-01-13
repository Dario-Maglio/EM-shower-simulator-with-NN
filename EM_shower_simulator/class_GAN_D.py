"""Conditional GAN structure and subroutines."""

# Examples of conditional GANs from which we built our neural network :
# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
# https://keras.io/examples/generative/conditional_gan/

import os
import time
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam

from IPython import display

#-------------------------------------------------------------------------------
"""Constant parameters of configuration and definition of global objects."""

# Configuration parameters
N_PID = 3
N_ENER = 30 + 1
NOISE_DIM = 1024
ENERGY_NORM = 6.503
ENERGY_SCALE = 1000000.

# Scale parameter for energy loss
PARAM_EN = 0.01

# Create a random seed, to be used during the evaluation of the cGAN.
tf.random.set_seed(42)
num_examples = 6
test_noise = [tf.random.normal([num_examples, NOISE_DIM]),
              tf.random.uniform([num_examples, 1], minval= 0., maxval=N_ENER),
              tf.random.uniform([num_examples, 1], minval= 0., maxval=N_PID)]

# Define logger
logger = logging.getLogger("CGANLogger")

#-------------------------------------------------------------------------------

def generator_loss(fake_output):
    """Definie generator loss:
    successes on fake samples from the generator valued as true samples by
    the discriminator fake_output.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    """Define discriminator loss:
    fails on fake samples (from generator) and successes on real samples.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def energy_loss(en_label, total_energy):
    #msle = tf.keras.losses.MeanSquaredLogarithmicError()
    msle = tf.keras.losses.MeanSquaredError()
    return msle(en_label, total_energy)

#-------------------------------------------------------------------------------

class ConditionalGAN(tf.keras.Model):
    """Class for a conditional GAN.
    It inherits keras.Model properties and functions.
    """
    def __init__(self, gener, discr, learning_rate=3e-5):
        """Constructor.
        Inputs:
        gener = generator network;
        discr = discriminator network.
        """
        super(ConditionalGAN, self).__init__()
        self.generator = gener
        self.discriminator = discr
        self.generator_optimizer = Adam(learning_rate)
        self.discriminator_optimizer = Adam(1e-4)
        self.gener_loss_tracker = Mean(name="generator_loss")
        self.discr_loss_tracker = Mean(name="discriminator_loss")
        self.energ_loss_tracker = Mean(name="energy_loss")

        # Create a manager to save rusults from training in form of checkpoints.
        self.checkpoint = tf.train.Checkpoint(
                           generator=self.generator,
                           discriminator=self.discriminator,
                           generator_optimizer=self.generator_optimizer,
                           discriminator_optimizer=self.discriminator_optimizer)

        self.manager = tf.train.CheckpointManager(self.checkpoint,
                                        './training_checkpoints', max_to_keep=3)

    @property
    def metrics(self):
        """Metrics of the cGAN network."""
        return [self.gener_loss_tracker, self.discr_loss_tracker]

    def summary(self):
        """Summary method for both the generator and discriminator."""
        print("\n\n Conditional GAN model summary:\n")
        self.generator.summary()
        print("\n")
        self.discriminator.summary()

    def plot_model(self):
        """Plot and saves the current cGAN model scheme."""
        save_path = Path('model_plot').resolve()
        if not os.path.isdir(save_path):
           os.makedirs(save_path)

        fig = plt.figure("Model scheme", figsize=(20,10))
        plt.subplot(1, 2, 1)
        plt.title("Generator")
        file_name = "cgan-generator.png"
        path = os.path.join(save_path, file_name)
        plot_model(self.generator, to_file=path, show_shapes=True)
        plt.imshow(imread(path))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Discriminator")
        file_name = "cgan-discriminator.png"
        path = os.path.join(save_path, file_name)
        plot_model(self.discriminator, to_file=path, show_shapes=True)
        plt.imshow(imread(path))
        plt.axis("off")
        plt.show()

        file_name = "cgan-scheme.png"
        path = os.path.join(save_path, file_name)
        fig.savefig(os.path.join(save_path, file_name))

    def generate_and_save_images(self, noise, epoch=0):
        """Generate images from the noise, plot and evaluate them."""
        # 1 - Generate images
        predictions = self.generator(noise, training=False)
        decisions = self.discriminator([predictions, noise[1]])
        logger.info(f"Shape of generated images: {predictions.shape}")

        # 2 - Plot the generated images
        k=0
        fig = plt.figure("Generated showers", figsize=(20,10))
        num_examples = predictions.shape[0]
        for i in range(num_examples):
           for j in range(predictions.shape[1]):
              k=k+1
              plt.subplot(num_examples, predictions.shape[1], k)
              plt.imshow(predictions[i,j,:,:,0]) #, cmap="gray")
              plt.axis("off")

        for example in range(len(noise[0]) ):
            print(f"{example+1}) Primary particle={int(noise[2][example][0])}"
                 +f"\nInitial energy ={noise[1][example][0]}"
                 +f"\tGenerated energy ={decisions[1][example]}"
                 +f"\tDecision ={decisions[0][example]}")
        plt.show()

        # 3 - Save the generated images
        save_path = Path('training_results').resolve()
        file_name = f"image_at_epoch_{epoch}.png"
        if not os.path.isdir(save_path):
           os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, file_name))

    def evaluate(self):
        """Return the generator from the last checkpoint and show examples."""
        try:
           self.checkpoint.restore(self.manager.latest_checkpoint)
        except:
           raise Exception('Error loading the model: corrupted checkpoint!')
        print("Showing example showers from noise.")
        self.generate_and_save_images(noise)
        return self.generator

    def compile(self):
        """Wrap the default compile method of the network.
        Quite useless in this case because the training set up has been done in
        the constructor of the class. It associate to the new abstract model an
        optimizer attribute 'rmsprop', and loss, metrics=None.
        """
        super(ConditionalGAN, self).compile()

    def scheduler(self, epoch, lr):
        """Not implemented yet"""
        if epoch < 34:
           return lr
        else:
           lr = lr * tf.math.exp(-0.1)
           print(f"Learning rate updated to {lr}")
           return lr

    def train_step(self, dataset):
        """Train step of the cGAN.
        Inputs:
        dataset = combined images  and labels upon which the network trained.

        Description:
        1) Create a noise to feed into the model for the images generation;
        2) Generate images and calculate losses using real images and labels;
        3) Calculate gradients using loss values and model variables;
        4) Process Gradients and Run the Optimizer.
        """
        real_images, en_labels, pid_labels = dataset

        noise = tf.random.normal([real_images.shape[0], NOISE_DIM])

        generator_input = [noise, en_labels, pid_labels]

        # GradientTape method records operations for automatic differentiation.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            generated_images = self.generator(generator_input, training=True)

            real_sample = [real_images, en_labels]#, pid_labels]
            real_output = self.discriminator(real_sample, training=True)

            fake_sample = [generated_images, en_labels]#, pid_labels]
            fake_output = self.discriminator(fake_sample, training=True)

            energ_loss = energy_loss(en_labels, fake_output[1])
            gener_loss = generator_loss(fake_output[0])
            discr_loss = discriminator_loss(real_output[0], fake_output[0])

            gener_total_loss = PARAM_EN * energ_loss + gener_loss
            discr_total_loss = discr_loss # + PARAM_EN * energ_loss

        grad_generator = gen_tape.gradient(gener_total_loss,
                                        self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grad_generator,
                                        self.generator.trainable_variables))

        grad_discriminator = disc_tape.gradient(discr_total_loss,
                                        self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grad_discriminator,
                                        self.discriminator.trainable_variables))

        # Monitor losses
        self.gener_loss_tracker.update_state(gener_loss)
        self.discr_loss_tracker.update_state(discr_loss)
        self.energ_loss_tracker.update_state(energ_loss)

        return{
            "gener_loss": self.gener_loss_tracker.result(),
            "discr_loss": self.discr_loss_tracker.result(),
            "energ_loss": self.energ_loss_tracker.result()
        }

    def fit(self, dataset, epochs=1, batch=32):
        """Wrap the default training function of the model."""
        dataset = dataset.batch(batch, drop_remainder=True)
        return super(ConditionalGAN, self).fit(dataset, epochs=epochs)

    def train(self, dataset, epochs=1, batch=32, verbose=1):
        """Define the training function of the cGAN.
        Inputs:
        dataset = combined real images vectors and labels;
        epochs = number of epochs for the training.

        For each epoch:
        1) For each batch of the dataset, run the custom "train_step" function;
        2) Produce images;
        3) Save the model every 5 epochs as a checkpoint;
        4) Print out the completed epoch no. and the time spent;
        5) Then generate a final image after the training is completed.
        """
        dataset = dataset.batch(batch, drop_remainder=True)

        # Call checkpoint manager to load the state or restart from scratch
        switch = input("Do you want to restore the last checkpoint? [y/N]")
        if switch=='y':
            if self.manager.latest_checkpoint:
                self.checkpoint.restore(self.manager.latest_checkpoint)
                print(f"Restored from {self.manager.latest_checkpoint}")
            else:
                print("No checkpoint found: initializing from scratch.")
        else:
            print("Initializing from scratch.")

        # Define default callbacks for the training
        callbacks =  tf.keras.callbacks.CallbackList([
                                                 tf.keras.callbacks.History()])
        callbacks.set_model(self)


        # Start training operations
        display.clear_output(wait=True)
        callbacks.on_train_begin()
        for epoch in range(epochs):
           print(f"Running EPOCH = {epoch + 1}/{epochs}")
           callbacks.on_epoch_begin(epoch)
           progbar = tf.keras.utils.Progbar(len(dataset), verbose=verbose)

           start = time.time()
           for index, image_batch in enumerate(dataset):
              callbacks.on_train_batch_begin(index)
              batch_logs = self.train_step(image_batch)
              status_to_display = zip(batch_logs.keys(), batch_logs.values())
              progbar.update(index, status_to_display)
              callbacks.on_train_batch_end(index, batch_logs)
           end = time.time() - start

           display.clear_output(wait=True)
           print(f"EPOCH = {epoch + 1}/{epochs}")
           for state in status_to_display:
               print(f"{state[0]} = {state[1]}")
           print (f"Time for epoch {epoch + 1} = {end} sec.")
           self.generate_and_save_images(test_noise, epoch + 1)
           #plt.figure("Evolution of losses per epochs")
           #plt.plot(self.history.history["gener_loss"], label="gener_loss")
           #plt.plot(self.history.history["discr_loss"], label="discr_loss")
           #plt.plot(self.history.history["energ_loss"], label="energ_loss")
           #plt.xlim(1, epochs)
           #plt.show()

           if (epoch + 1) % 5 == 0:
              save_path = self.manager.save()
              print(f"Saved checkpoint for epoch {epoch + 1}: {save_path}")

           callbacks.on_epoch_end(epoch, batch_logs)
        callbacks.on_train_end(batch_logs)
        return self.history
