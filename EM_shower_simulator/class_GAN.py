"""Conditional GAN structure and subroutines."""

# Examples of conditional GANs from which we built our neural network :
# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
# https://keras.io/examples/generative/conditional_gan/

import sys
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
#from constants import *
ENERGY_NORM = 6.7404
ENERGY_SCALE = 1000000

N_PID = 3                               # number of pid classes
N_ENER = 30 + 1                         # number of en classes
PARAM_EN = 0.01                         # parameter in energy losses computation
NOISE_DIM = 2048
BUFFER_SIZE = 10400

MBSTD_GROUP_SIZE = 8                    #minibatch dimension


# Create a random seed, to be used during the evaluation of the cGAN.
tf.random.set_seed(12)
num_examples = 6
test_noise = [tf.random.normal([num_examples, NOISE_DIM]),
              tf.random.uniform([num_examples, 1], minval= 0., maxval=N_ENER),
              tf.random.uniform([num_examples, 1], minval= 0., maxval=N_PID)]

# Define logger
logger = logging.getLogger("CGANLogger")

#-------------------------------------------------------------------------------

@tf.function
def compute_energy(in_images):
    """Compute energy deposited in detector."""
    in_images = tf.cast(in_images, tf.float32)

    en_images = tf.math.multiply(in_images, ENERGY_NORM)
    en_images = tf.math.pow(10., en_images)
    en_images = tf.math.divide(en_images, ENERGY_SCALE)
    en_images = tf.math.reduce_sum(en_images, axis=[1,2,3])
    return en_images

@tf.function
def generator_loss(fake_output):
    """Definie generator loss:
    successes on fake samples from the generator valued as true samples by
    the discriminator fake_output.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def discriminator_loss(real_output, fake_output):
    """Define discriminator loss:
    fails on fake samples (from generator) and successes on real samples.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

@tf.function
def energy_loss(en_label, energy):
    #msle = tf.keras.losses.MeanSquaredLogarithmicError()
    msle = tf.keras.losses.MeanSquaredError()
    return msle(en_label, energy) * PARAM_EN

@tf.function
def particle_loss(pid_labels, fake_labels):
    labels = tf.math.add(pid_labels, -1)
    labels = tf.math.abs(labels)
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(labels, fake_labels) * PARAM_EN

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

        # Metrics
        self.gener_loss_tracker = Mean(name="generator_loss")
        self.discr_loss_tracker = Mean(name="discriminator_loss")
        self.energ_loss_tracker = Mean(name="energy_loss")
        self.parID_loss_tracker = Mean(name="particle_loss")
        self.label_loss_tracker = Mean(name="label_loss")

        # Scheduler attributes and optimizers
        self.generator_optimizer = Adam(learning_rate * 10)
        self.discriminator_optimizer = Adam(learning_rate)

        # Manager to save rusults from training in form of checkpoints.
        self.history = {}
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
        return [self.gener_loss_tracker,
                self.discr_loss_tracker,
                self.energ_loss_tracker,
                self.parID_loss_tracker,
                self.label_loss_tracker]

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

    def compile(self):
        """Compile method of the cGAN network.
        Quite useless in this case because the training set up has been done in
        the constructor of the class. It associate to the new abstract model an
        optimizer attribute 'rmsprop', and loss, metrics=None.
        """
        super(ConditionalGAN, self).compile()

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
            print(f"{example+1}) Primary particle = {int(noise[2][example][0])}"
                 +f"   Predicted particle = {decisions[2][example][0]}"
                 +f"\nInitial energy = {noise[1][example][0]}   "
                 +f"Generated energy = {energies[example][0]}   "
                 +f"Predicted energy = {decisions[1][example][0]}   "
                 +f"Decision = {decisions[0][example][0]}")
        plt.show()

        # 3 - Save the generated images
        save_path = Path('training_results').resolve()
        file_name = f"image_at_epoch_{epoch}.png"
        if not os.path.isdir(save_path):
           os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, file_name))

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

    def scheduler(self, epoch, logs, wake_up):
        """Starting from epoch S_EPOCH, the scheduler boosts the generator or
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

        """
        if (epoch == wake_up):
           learning_rate_pro = self.generator_optimizer.lr * 3.
           if (logs["gener_loss"] > logs["discr_loss"]):
              self.generator_optimizer.lr = learning_rate_pro
              self.switch = False
              print("Power to the generator!")
           elif (logs["gener_loss"] < logs["discr_loss"]):
              self.discriminator_optimizer.lr = learning_rate_pro
              self.switch = True
              print("Power to the discriminator!")
        elif (epoch > wake_up):
           decrease = 0.999
           gener_lr = self.generator_optimizer.lr.numpy()
           discr_lr = self.discriminator_optimizer.lr.numpy()
           self.learning_rate = self.learning_rate * decrease
           self.generator_optimizer.lr = gener_lr * decrease
           self.discriminator_optimizer.lr = discr_lr * decrease
           logger.info(f"Learning rate setted to {self.learning_rate}.")
           if (logs["gener_loss"] > logs["discr_loss"]) & self.switch:
              discr_lr = self.discriminator_optimizer.lr.numpy()
              self.discriminator_optimizer.lr = self.learning_rate
              self.generator_optimizer.lr = discr_lr
              self.switch = False
              print(f"Learning rate switched: power to the generator!")
           elif (logs["gener_loss"] < logs["discr_loss"]) & (not self.switch):
              gener_lr = self.generator_optimizer.lr.numpy()
              self.generator_optimizer.lr = self.learning_rate
              self.discriminator_optimizer.lr = gener_lr
              self.switch = True
              print(f"Learning rate switched: power to the discriminator!")
        """

    @tf.function
    def generate_noise(self, num_examples=num_examples):
        """Generate a set of num_examples noise inputs for the generator."""
        return [tf.random.normal([num_examples, NOISE_DIM]),
                tf.random.uniform([num_examples, 1], minval= 0., maxval=N_ENER),
                tf.random.uniform([num_examples, 1], minval= 0., maxval=N_PID)]

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

        noise = self.generate_noise(num_examples=real_images.shape[0])[0]

        generator_input = [noise, en_labels, pid_labels]

        # GradientTape method records operations for automatic differentiation.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            generated_images = self.generator(generator_input, training=True)

            energies = compute_energy(generated_images)
            # print("\nREAL IMAGES")
            real_output = self.discriminator(real_images, training=True)
            # print("*************************************************************")
            # print("\nGENERATED IMAGES")
            # print(generated_images)
            fake_output = self.discriminator(generated_images, training=True)
            # print("*************************************************************")

            gener_loss = generator_loss(fake_output[0])
            discr_loss = discriminator_loss(real_output[0], fake_output[0])

            label_loss = energy_loss(en_labels, energies)
            fake_energ_loss = energy_loss(en_labels, fake_output[1])
            real_energ_loss = energy_loss(en_labels, real_output[1])
            fake_parID_loss = particle_loss(pid_labels, fake_output[2])
            real_parID_loss = particle_loss(pid_labels, real_output[2])

            gener_total_loss = gener_loss + label_loss + fake_energ_loss + fake_parID_loss
            discr_total_loss = discr_loss + real_energ_loss + real_parID_loss

            if np.isnan(label_loss):
                print(generated_images)
                raise AssertionError("ERROR IN COMPUTING LABEL LOSS: NAN VALUE")
            elif np.isnan(fake_energ_loss):
                raise AssertionError("ERROR IN COMPUTING FAKE ENERG LOSS: NAN VALUE")
            elif np.isnan(real_energ_loss):
                raise AssertionError("ERROR IN COMPUTING REAL ENERG LOSS: NAN VALUE")
            elif np.isnan(fake_parID_loss):
                raise AssertionError("ERROR IN COMPUTING FAKE PARID LOSS: NAN VALUE")
            elif np.isnan(real_parID_loss):
                raise AssertionError("ERROR IN COMPUTING REAL PARID LOSS: NAN VALUE")
            elif np.isnan(gener_loss):
                raise AssertionError("ERROR IN COMPUTING GENER LOSS: NAN VALUE")
            elif np.isnan(discr_loss) :
                raise AssertionError("ERROR IN COMPUTING DISCR LOSS: NAN VALUE")

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
        self.energ_loss_tracker.update_state(real_energ_loss)
        self.parID_loss_tracker.update_state(real_parID_loss)
        self.label_loss_tracker.update_state(label_loss)

        if np.isnan(label_loss):
            raise AssertionError("ERROR IN MINIMIZATION LABEL LOSS: NAN VALUE")
        elif np.isnan(fake_energ_loss):
            raise AssertionError("ERROR IN MINIMIZATION FAKE ENERG LOSS: NAN VALUE")
        elif np.isnan(real_energ_loss):
            raise AssertionError("ERROR IN MINIMIZATION REAL ENERG LOSS: NAN VALUE")
        elif np.isnan(fake_parID_loss):
            raise AssertionError("ERROR IN MINIMIZATION FAKE PARID LOSS: NAN VALUE")
        elif np.isnan(real_parID_loss):
            raise AssertionError("ERROR IN MINIMIZATION REAL PARID LOSS: NAN VALUE")
        elif np.isnan(gener_loss):
            raise AssertionError("ERROR IN MINIMIZATION GENER LOSS: NAN VALUE")
        elif np.isnan(discr_loss) :
            raise AssertionError("ERROR IN MINIMIZATION DISCR LOSS: NAN VALUE")

        return{
            "gener_loss": self.gener_loss_tracker.result(),
            "discr_loss": self.discr_loss_tracker.result(),
            "energ_loss": self.energ_loss_tracker.result(),
            "parID_loss": self.parID_loss_tracker.result(),
            "label_loss": self.label_loss_tracker.result()
        }

    def fit(self, dataset, epochs=1, batch=32):
        """Wrap the default training function of the model."""
        dataset = dataset.batch(batch, drop_remainder=True)
        return super(ConditionalGAN, self).fit(dataset, epochs=epochs)

    def train(self, dataset, epochs=1, batch=32, wake_up=10, verbose=1):
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
           progbar = tf.keras.utils.Progbar(len(dataset), verbose=1)

           # Start iterate on batches
           start = time.time()
           for index, image_batch in enumerate(dataset):
              try:
                  logs = self.train_step(image_batch)
              except AssertionError as error:
                  print(f"Epoch {epoch}, batch {index}: {error}")
                  sys.exit()
              progbar.update(index, zip(logs.keys(), logs.values()))
           end = time.time() - start

           # Dispaly results and save images
           display.clear_output(wait=True)
           print(f"EPOCH = {epoch + 1}/{epochs}")
           for log in logs:
               print(f"{log} = {logs[log]}")
           print (f"Time for epoch {epoch + 1} = {end} sec.")
           self.generate_and_save_images(test_noise, epoch + 1)

           # Save checkpoint
           if (epoch + 1) % 3 == 0:
              save_path = self.manager.save()
              print(f"Saved checkpoint for epoch {epoch + 1}: {save_path}")

           # Update history and call the scheduler
           for key, value in logs.items():
               self.history.setdefault(key, []).append(value)
           self.scheduler(epoch + 1, logs, wake_up=wake_up)
        return self.history
