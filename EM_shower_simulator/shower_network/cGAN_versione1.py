""" Train the GAN with the Geant generated dataset and save the model """

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
from tensorflow.keras.utils import to_categorical
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import uproot as up


BUFFER_SIZE = 1000
BATCH_SIZE = 128
NOISE_DIM = 1000
N_CLASSES = 1

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# creo cartella in cui salvo i risultati del train come checkpoint. Inoltro genero
# un seed random costante, che riutilizzo dopo

EPOCHS = 100
num_examples_to_generate = 5
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

generator_optimizer = tf.keras.optimizers.Adam(3e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(3e-4)

#-------------------------------------------------------------------------------

#path=pl.Path('Dataset/Filtered_data/data_MVA.root')
#tpath = pathlib.Path(path).resolve()
file = up.open("/content/EM-shower-simulator-with-NN/Dataset/Filtered_data/data_MVA.root")
branches = file['h'].arrays()

train_images = np.array(branches["shower"]).astype("float32")
#images in log10 scale, zeros like pixel are set to -5, maximum of pixel is 2.4E5 keV ≃ 5.4 in log scale
train_images = (train_images -0.2) / 5.2 # Normalize the images to [-1, 1]
train_images = np.reshape(train_images, (-1, 12, 12, 12, 1))

en_label = np.array(branches["en_in"]).astype("float32")/2000000.0 # np.zeros(1000).astype("float32")
#pid_label = np.array(branches["primary"])
train_labels = np.transpose(en_label)
train_labels = np.reshape(train_labels, (1000,1))

print(train_labels.shape)
# Batch and shuffle the data
train_dataset = ( tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                .shuffle(BUFFER_SIZE).batch(BATCH_SIZE) )

def debug_shower():
    """
    Genera dei cerchi random e li visualizza
    """
    print(train_images.shape)
    #plt.figure(figsize=(20,150))
    k=0
    N_showers = 10
    for i in range(N_showers):
        for j in range(12):
            k=k+1
            plt.subplot(N_showers, 12, k)
            plt.imshow(train_images[i, j, :, :, 0], cmap='gray')
            plt.axis("off")
    plt.show()
#-------------------------------------------------------------------------------

def make_generator_model():
    """
    Definisce il modello del generatore: Sequential con dei layer convolutivi Transpose
    """
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(N_CLASSES, 50)(in_label)
    # linear multiplication
    n_nodes = 3 * 3 * 3 *N_CLASSES
    li = Dense(n_nodes)(in_label)
    # reshape to additional channel
    li = Reshape((3, 3, 3, N_CLASSES))(li)
    # image generator input
    in_lat = Input(shape=(NOISE_DIM,))
    # foundation for 12x12x12 image
    n_nodes = 256 * 3 * 3 * 3
    gen = Dense(n_nodes, use_bias=False)(in_lat)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((3, 3, 3, 256))(gen)
    #assert gen.shape == (None, 3, 3, 3, 256) # Note: None is the batch size

    # merge image gen and label input
    merge = Concatenate()([gen, li])

    gen = Conv3DTranspose(128, (5, 5, 5), strides=(1, 1, 1), padding='same', use_bias=False)(merge)
    #assert gen.shape == (None, 3, 3, 3, 128)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv3DTranspose(64, (5, 5, 5), strides=(2, 2, 2), padding='same', use_bias=False)(gen)
    #assert gen.shape == (None, 6, 6, 6, 64)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    output = (Conv3DTranspose(1, (5, 5, 5), strides=(2, 2, 2), padding='same',
                              use_bias=False, activation='tanh')(gen))
    #assert output.shape == (None, 12, 12, 12, 1)

    model = Model([in_lat, in_label], output)
    return model


def generator_loss(fake_output):
    """
    Definisce la loss del generatore: tengo conto dei successi sugli esempi fake(generatore)
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = make_generator_model()
plot_model(generator, to_file='generator.png')
Image("generator.png")
#-------------------------------------------------------------------------------

def debug_generator():
    """
    Create a random noise and generate a sample
    """
    generator = make_generator_model()
    noise = tf.random.normal([1, NOISE_DIM])
    labels = np.random.rand(N_CLASSES)
    generated_image = generator([noise, labels], training=False)
    print(generated_image.shape)

    plt.figure(figsize=(20,150))
    for j in range(12):
        plt.subplot(1, 12, j+1)
        plt.imshow(generated_image[0,j,:,:,0], cmap='gray')
        plt.axis("off")
    plt.show()

#-------------------------------------------------------------------------------

def make_discriminator_model():
    """
    Definisce il modello del discriminator: Sequential con dei layer convolutivi
    """
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(N_CLASSES, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = 12*12*12
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((12,12,12, 1))(li)
    # image input
    in_image = Input(shape=(12,12,12,1))
    # concat label as a channel
    merge = Concatenate()([in_image, li])
    # downsample

    discr = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding='same')(merge)
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    discr = Conv3D(64, (5, 5, 5), strides=(2, 2, 2), padding='same')(discr)
    discr = LeakyReLU()(discr)
    discr = Dropout(0.3)(discr)

    discr = Flatten()(discr)
    output = Dense(1, activation="sigmoid")(discr)

    model = Model([in_image, in_label], output)
    return model

def discriminator_loss(real_output, fake_output):
    """
    Definisce la loss del discrimatore: tengo conto dei fail sugli esempi fake(generatore)
    e dei successi sugli esempi veri
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

discriminator = make_discriminator_model()
plot_model(discriminator, to_file='discriminator.png')
Image("discriminator.png")
#-------------------------------------------------------------------------------

def debug_discriminator():
    """
    Verifico output del discriminatore sul sample generato random
    """
    discriminator = make_discriminator_model()
    noise = tf.random.normal([1, NOISE_DIM])
    labels = np.random.rand(N_CLASSES)
    print(labels.shape)
    generator = make_generator_model()
    generated_image = generator([noise,labels], training=False)
    decision = discriminator( [generated_image,labels] )
    print(decision)

#-------------------------------------------------------------------------------

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_and_save_images(model, epoch, test_input):
    """
    Genera e salva immagini ad ogni epoca
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
        plt.imshow(predictions[i,j,:,:,0] * 127.5 + 127.5, cmap='gray')
        plt.axis("off")
    plt.show()
    # 3 - Save the generated images
    plt.savefig(f'image_at_epoch_{epoch}.png')
    plt.show()

#-------------------------------------------------------------------------------

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.generator_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.discr_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.discr_loss_tracker]

    def compile(self):
        super(ConditionalGAN, self).compile()

    # tf.function annotation causes the function
    # to be "compiled" as part of the training

    def train_step(self, dataset):
        """
        1 - Create a random noise to feed it into the model for the image generation ;
        2 - Generate images and calculate loss values ;
        3 - Calculate gradients using loss values and model variables;
        4 - Process Gradients and Run the Optimizer ;
        """
        real_images, real_labels = dataset

        #dummy_labels = real_labels[:, :, None, None, None]
        #dummy_labels = tf.repeat( dummy_labels, repeats=[12 * 12 * 12] )
        #dummy_labels = tf.reshape( dummy_labels, (-1, 12 ,12 , 12, N_CLASSES) )
        #print(dummy_labels.shape)
        random_noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        #noise_and_labels= tf.concat([random_noise, real_labels],axis=1)

        # GradientTape method records operations for automatic differentiation.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            generated_images = self.generator([random_noise, real_labels], training=True)

            #fake_image_and_labels = tf.concat([generated_images, dummy_labels], -1)
            #real_image_and_labels = tf.concat([real_images, dummy_labels], -1)

            real_output = self.discriminator([real_images, real_labels], training=True)
            fake_output = self.discriminator([generated_images, real_labels], training=True)

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
      Definisce la funzione vera e propria di train della rete GAN :
      For each epoch:
      -) For each batch of the epoch, run the custom "train_step" function;
      -) Produce images;
      -) Save the model every 5 epochs as a checkpoint;
      -) Print out the completed epoch no. and the time spent;
      Then generate a final image after the training is completed.
      """
      for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            self.train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(self.generator,
                                epoch + 1,
                                seed)

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print (f'Time for epoch {epoch + 1} is {time.time()-start} sec')

      display.clear_output(wait=True)
      generate_and_save_images(self.generator,
                             epochs,
                             seed)

#-------------------------------------------------------------------------------
if __name__=="__main__":
    debug_shower()
    debug_generator();
    debug_discriminator();

    cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=NOISE_DIM)
    cond_gan.compile()
    cond_gan.fit(train_dataset, epochs=EPOCHS)
    #cond_gan.train(train_dataset, EPOCHS)

    """
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

    # evaluate the model
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    """

    # save model and architecture to single file
    cond_gan.save("model.h5")
    print("Model saved to disk")
