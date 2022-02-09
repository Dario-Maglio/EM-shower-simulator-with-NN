""" Conditional GAN Class and structure """

import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.utils import Progbar
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.train import CheckpointManager as Manager

from IPython import display

from unbiased_metrics import shower_depth_lateral_width

#-------------------------------------------------------------------------------
"""Constant parameters of configuration and definition of global objects."""

# Configuration parameters
#from constants import *
ENERGY_NORM = 6.7404
ENERGY_SCALE = 1000000

N_PID = 3                               # number of pid classes
N_ENER = 30 + 1                         # number of en classes
PARAM_EN = 0.01                         # parameter in energy losses computation
NOISE_DIM = 2048

# Create a random seed, to be used during the evaluation of the cGAN.
tf.random.set_seed(42)
num_examples = 6
test_noise = [tf.random.normal([num_examples, NOISE_DIM]),
              tf.random.uniform([num_examples, 1], minval= 0., maxval=N_ENER),
              tf.random.uniform([num_examples, 1], minval= 0., maxval=N_PID)]

# Define logger
logger = logging.getLogger("CGANLogger")

#-------------------------------------------------------------------------------

def metrics_control(metrics, string = "MINIMIZATION"):
    """Prevent the training to use continue when a NaN is found."""
    for key in metrics:
        if np.isnan(metrics[key]):
            raise AssertionError(f"\nERROR IN {string} {key}: NAN VALUE")

@tf.function
def compute_energy(in_images):
    """Compute energy deposited into the detector."""
    in_images = tf.cast(in_images, tf.float32)
    en_images = tf.math.multiply(in_images, ENERGY_NORM)
    en_images = tf.math.pow(10., en_images)
    en_images = tf.math.divide(en_images, ENERGY_SCALE)
    en_images = tf.math.reduce_sum(en_images, axis=[1,2,3])
    return en_images

#-------------------------------------------------------------------------------

class ConditionalGAN(tf.keras.Model):
    """Class for a conditional GAN.
    It inherits keras.Model properties and functions.
    """
    def __init__(self, gener, discr, learning_rate=2e-5):
        """Constructor.
        Inputs:
        gener = generator network;
        discr = discriminator network;
        learning_rate = starting poor learning rate.
        """
        super(ConditionalGAN, self).__init__()
        self.generator = gener
        self.discriminator = discr
        self.history = {}
        self.logs = {}

        # Metrics
        self.gener_loss_tracker = Mean(name="gener_loss")
        self.discr_loss_tracker = Mean(name="discr_loss")
        self.energ_loss_tracker = Mean(name="energy_loss")
        self.parID_loss_tracker = Mean(name="particle_loss")
        self.computed_e_tracker = Mean(name="computed_loss")

        # Unbiased metrics
        self.mean_depth_tracker = Mean(name="mean_depth")
        self.std_depth_tracker  = Mean(name="std_depth")
        self.mean_lateral_tracker = Mean(name="mean_lateral")
        self.std_lateral_tracker  = Mean(name="std_lateral")

        # Optimizers
        self.generator_optimizer = Adam(learning_rate * 10)
        self.discriminator_optimizer = Adam(learning_rate)

        # Manager to save rusults from training in form of checkpoints
        self.checkpoint = tf.train.Checkpoint(
                           generator=self.generator,
                           discriminator=self.discriminator,
                           generator_optimizer=self.generator_optimizer,
                           discriminator_optimizer=self.discriminator_optimizer)

        self.manager = Manager(self.checkpoint, './checkpoints', max_to_keep=5)

    @property
    def metrics(self):
        """Metrics of the cGAN network."""
        return [self.gener_loss_tracker,
                self.discr_loss_tracker,
                self.energ_loss_tracker,
                self.parID_loss_tracker,
                self.computed_e_tracker]

    def compile(self):
        """Compile method of the cGAN network.
        Quite useless in this case because the training set up has been done in
        the constructor of the class. It associate to the new abstract model an
        optimizer attribute 'rmsprop', and loss, metrics=None.
        """
        super(ConditionalGAN, self).compile()

    def summary(self):
        """Summary method of the cGAN network."""
        print("\nPrinting conditional GAN summary to file.\n")
        save_path = Path('model_plot').resolve()
        if not os.path.isdir(save_path):
           os.makedirs(save_path)
        file_name = "cgan-summary.txt"
        path = os.path.join(save_path, file_name)
        with open(path, 'w') as file:
           file.write('\nConditional GAN summary\n\n')
           self.generator.summary(print_fn=lambda x: file.write(x + '\n'))
           file.write('\n\n')
           self.discriminator.summary(print_fn=lambda x: file.write(x + '\n'))
           file.write('\n\n')

    def plot_model(self):
        """Plot_model method of the cGAN network."""
        print("\nPlotting and saving conditional GAN scheme.\n")
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

        file_name = "cgan-scheme.png"
        path = os.path.join(save_path, file_name)
        fig.savefig(os.path.join(save_path, file_name))
        plt.close()

    def generate_noise(self, num_examples=num_examples):
        """Generate a set of num_examples noise inputs for the generator."""
        return [tf.random.normal([num_examples, NOISE_DIM]),
                tf.random.uniform([num_examples, 1], minval= 0., maxval=N_ENER),
                tf.random.uniform([num_examples, 1], minval= 0., maxval=N_PID)]

    def evaluate(self, num_examples=num_examples):
        """Restore the last checkpoint and return the models."""
        if self.manager.latest_checkpoint:
            latest_check = self.manager.latest_checkpoint
            try:
               self.checkpoint.restore(latest_check).expect_partial()
            except:
               raise Exception("Invalid checkpoint.")
            print(f"Restored from {latest_check}")
            return self.generator, self.discriminator
        else:
            raise Exception("No checkpoint found.")

    def generate_and_save_images(self, noise, epoch=0):
        """Use the current status of the NN to generate images from the noise,
        plot, evaluate and save them.
        Inputs:
        noise = noise with the generator input shape.
        """
        # 1 - Generate images
        predictions = self.generator(noise, training=False)
        decisions = self.discriminator(predictions, training=False)
        logger.info(f"Shape of generated images: {predictions.shape}")
        energies = compute_energy(predictions)

        # 2 - Plot the generated images
        k = 0
        num_examples = predictions.shape[0]
        fig = plt.figure("Generated showers", figsize=(20,10))
        for i in range(num_examples):
            print(f"Example {i+1}\t"
                 +f"Primary particle = {int(noise[2][i][0])}\t"
                 +f"Predicted particle = {decisions[2][i][0]}\n"
                 +f"Initial energy = {noise[1][i][0]}\t"
                 +f"Generated energy = {energies[i][0]}\t"
                 +f"Predicted energy = {decisions[1][i][0]}\t"
                 +f"Decision = {decisions[0][i][0]}\n")
            for j in range(predictions.shape[1]):
                k=k+1
                plt.subplot(num_examples, predictions.shape[1], k)
                plt.imshow(predictions[i,j,:,:,0]) #, cmap="gray")
                plt.axis("off")
        plt.show()

        # 3 - Save the generated images
        save_path = Path('model_results').resolve()
        file_name = f"image_at_epoch_{epoch}.png"
        if not os.path.isdir(save_path):
           os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, file_name))

    def scheduler(self, epoch, logs, wake_up):
        """Decrease the learning_rate:
        Starting from epoch wake_up, the scheduler boosts the generator or
        discriminator learning rate depending on which is doing better. The
        comparison is made looking at the losses stored in logs.
        """
        if (epoch > wake_up):
           decrease = 0.999
           gener_lr = self.generator_optimizer.lr.numpy()
           discr_lr = self.discriminator_optimizer.lr.numpy()
           self.generator_optimizer.lr = gener_lr * decrease
           self.discriminator_optimizer.lr = discr_lr * decrease
           logger.info(f"Gener learning rate setted to {gener_lr * decrease}.")
           logger.info(f"Discr learning rate setted to {discr_lr * decrease}.")

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
        mean_squared = MeanSquaredError()
        cross_entropy = BinaryCrossentropy()

        real_images, en_labels, pid_labels = dataset
        noise = self.generate_noise(num_examples=real_images.shape[0])[0]

        # GradientTape method records operations for automatic differentiation.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Compute real and fake outputs
            generator_input = [noise, en_labels, pid_labels]
            generated_images = self.generator(generator_input, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Compute GAN loss on decisions
            ones = tf.ones_like(real_output[0])
            zero = tf.zeros_like(fake_output[0])
            real_loss = cross_entropy(ones, real_output[0])
            fake_loss = cross_entropy(zero, fake_output[0])

            gener_loss = cross_entropy(ones, fake_output[0])
            discr_loss = real_loss + fake_loss

            # Generated and computed energies
            energies = compute_energy(generated_images)
            computed_e = mean_squared(en_labels, energies)

            # Compute auxiliary energy and particle losses
            fake_energ = mean_squared(en_labels, fake_output[1])  # or energies?
            real_energ = mean_squared(en_labels, real_output[1])

            parID = tf.math.abs(tf.math.add(pid_labels, -1))
            fake_parID = cross_entropy(parID, fake_output[2])
            real_parID = cross_entropy(parID, real_output[2])

            aux_gener_loss = (fake_energ + computed_e) * PARAM_EN + fake_parID
            aux_discr_loss = (real_energ) * PARAM_EN + real_parID

            # Compute total losses
            gener_total_loss = aux_gener_loss + gener_loss
            discr_total_loss = aux_discr_loss + discr_loss

        self.logs = {"gener_loss":gener_loss, "discr_loss":discr_loss,
                "energy_loss":real_energ, "particle_loss":real_parID,
                "computed_loss":computed_e }
        metrics_control(self.logs, "COMPUTING")

        grad_generator = gen_tape.gradient(gener_total_loss,
                                        self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grad_generator,
                                        self.generator.trainable_variables))

        grad_discriminator = disc_tape.gradient(discr_total_loss,
                                        self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grad_discriminator,
                                        self.discriminator.trainable_variables))

        # Update metrics
        self.gener_loss_tracker.update_state(gener_loss)
        self.discr_loss_tracker.update_state(discr_loss)
        self.energ_loss_tracker.update_state(real_energ)
        self.parID_loss_tracker.update_state(real_parID)
        self.computed_e_tracker.update_state(computed_e)

        for element in self.metrics:
            self.logs[element.name] = element.result()

        # Prevent NaN propagation
        metrics_control(self.logs, "MINIMIZATION")

        return self.logs

    def train(self, dataset, epochs=1, batch=32, wake_up=100, verbose=1):
        """Define the training function of the cGAN.
        Inputs:
        dataset = combined real images vectors and labels;
        epochs = number of epochs for the training;
        batch = number of batch in which dataset must be split;
        wake_up = epoch in which learning rates start to switch and decrease.

        For each epoch:
        1) For each batch of the dataset, run the custom "train_step" function;
        2) Produce images;
        3) Save the model every 5 epochs as a checkpoint;
        4) Print out the completed epoch no. and the time spent;
        5) Then generate a final image after the training is completed.
        """
        if verbose :
            logger.setLevel(logging.DEBUG)
            logger.info('Logging level set on DEBUG.')
        else:
            logger.setLevel(logging.WARNING)
            logger.info('Logging level set on WARNING.')
        dataset = dataset.batch(batch, drop_remainder=True)

        # Call checkpoint manager to load the state or restart from scratch
        switch = input("Do you want to restore the last checkpoint? [y/N]")
        if switch=='y':
           if self.manager.latest_checkpoint:
              latest_check = self.manager.latest_checkpoint
              try:
                 self.checkpoint.restore(latest_check).expect_partial()
                 print(f"Restored from {latest_check}")
              except:
                 print("Invalid checkpoint: init from scratch.")
           else:
              print("No checkpoint found: initializing from scratch.")
        else:
            print("Initializing from scratch.")

        # Start training operations
        display.clear_output(wait=True)
        for epoch in range(epochs):
           print(f"Running EPOCH = {epoch + 1}/{epochs}")

           # Define the progbar
           progbar = Progbar(len(dataset), verbose=1)

           # Start iterate on batches
           start = time.time()
           for index, image_batch in enumerate(dataset):
              try:
                  self.train_step(image_batch)
              except AssertionError as error:
                  print(f"\nEpoch {epoch}, batch {index}: {error}")
                  sys.exit()
              progbar.update(index, zip(self.logs.keys(), self.logs.values()))
              if(index==0):
                  unb_metr = shower_depth_lateral_width(image_batch[0])
                  self.mean_depth_tracker.update_state(unb_metr["shower mean depth"])
                  self.std_depth_tracker.update_state(unb_metr["shower std depth"])
                  self.mean_lateral_tracker.update_state(unb_metr["mean lateral width"])
                  self.std_lateral_tracker.update_state(unb_metr["std lateral width"])

           end = time.time() - start

           # Dispaly results and save images
           display.clear_output(wait=True)
           print(f"EPOCH = {epoch + 1}/{epochs}")
           for log in self.logs:
               print(f"{log} = {self.logs[log]}")
           print (f"Time for epoch {epoch + 1} = {end} sec.\n")
           self.generate_and_save_images(test_noise, epoch + 1)

           # Update history and call scheduler
           for key, value in self.logs.items():
               self.history.setdefault(key, []).append(value)
           self.scheduler(epoch + 1, self.logs, wake_up=wake_up)

           # Save checkpoint
           if (epoch + 1) % 3 == 0:
              save_path = self.manager.save()
              print(f"Saved checkpoint for epoch {epoch + 1}: {save_path}")

        return self.history

    def fit(self, dataset, epochs=1, batch=32):
        """Wrap the default training function of the model."""
        dataset = dataset.batch(batch, drop_remainder=True)
        return super(ConditionalGAN, self).fit(dataset, epochs=epochs)
