"""
Esempio di GAN per simulare cerchi
"""
import os
import time

from IPython import display # A command shell for interactive computing in Python.
from tensorflow.keras.layers import (Dense,
                                     BatchNormalization,
                                     LeakyReLU,
                                     Reshape,
                                     Conv3DTranspose,
                                     Conv3D,
                                     Dropout,
                                     Flatten)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

nsamples=50000
#-------------------------------------------------------------------------------

def background():
    """
    Definisce il background e le dimensioni dell'immagine
    """
    return np.zeros((12,12,1), np.uint8)

#-------------------------------------------------------------------------------

def randomColor():
    """
    Definisce il colore del cerchio
    """
    return (int(np.random.rand()*128+128))#,int(np.random.rand()*128+128),
                                          # int(np.random.rand()*128+128))
#-------------------------------------------------------------------------------

def drawCircle(color,x,y,radius):
    """
    Disegna un cerchio con le seguenti caratteristiche :
    -) c = colore ;
    -) x = posizione x del centro ;
    -) y = posizione y del centro ;
    -) r = raggio del cerchio ;
    """
    img = background()
    cv2.circle(img,(x,y),radius,color, -1)
    return img,x-radius,y-radius,x+radius,y+radius   #return image and bounding box

#-------------------------------------------------------------------------------

def genCircle():
    """"
    Genera e disegna un cerchio random
    """
    return drawCircle( randomColor(), int(np.random.rand()*2)+5,
                      int(np.random.rand()*2)+5, int(np.random.rand()*2)+2 )
#-------------------------------------------------------------------------------
#Genero le immagini
train_images=np.array( [[genCircle()[0] for l in range(12)]#if targets[x]# else genRectangle()[0]
                for x in range(nsamples)] )

#-------------------------------------------------------------------------------

def debug_circle():
    """
    Genera dei cerchi random e li visualizza
    """
    print(train_images.shape)
    plt.figure(figsize=(20,150))
    k=0
    for i in range(1):
        for j in range(12):
            k=k+1
            plt.subplot(1, 12, k)
            plt.imshow(train_images[i,j,:,:,0], cmap='gray')
            plt.axis("off")
    plt.show()


#-------------------------------------------------------------------------------
train_images = (train_images.reshape(train_images.shape[0], 12, 12, 12, 1)
                .astype('float32'))
train_images = (train_images -127.5) / 127.5 # Normalize the images to [-1, 1]

#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_DIM = 1000

# Batch and shuffle the data
train_dataset = ( tf.data.Dataset.from_tensor_slices(train_images)
                .shuffle(BUFFER_SIZE).batch(BATCH_SIZE) )

#-------------------------------------------------------------------------------

def make_generator_model():
    """
    Definisce il modello del generatore: Sequential con dei layer convolutivi Transpose
    """
    model = tf.keras.Sequential()
    model.add(Dense(3*3*3*256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((3, 3, 3, 256)))
    assert model.output_shape == (None, 3, 3, 3, 256) # Note: None is the batch size

    model.add(Conv3DTranspose(128, (5, 5, 5), strides=(1, 1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 3, 3, 3, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv3DTranspose(64, (5, 5, 5), strides=(2, 2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 6, 6, 6, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv3DTranspose(1, (5, 5, 5), strides=(2, 2, 2), padding='same',
                              use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 12, 12, 12, 1)

    return model

#-------------------------------------------------------------------------------

generator = make_generator_model()

#-------------------------------------------------------------------------------

def debug_generator():
    """
    Create a random noise and generate a sample
    """
    noise = tf.random.normal([1, NOISE_DIM])
    generated_image = generator(noise, training=False)
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

    model = tf.keras.Sequential()

    model.add(Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding='same', input_shape=[12, 12, 12, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv3D(64, (5, 5, 5), strides=(2, 2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model

#-------------------------------------------------------------------------------

discriminator = make_discriminator_model()

#-------------------------------------------------------------------------------

def debug_discriminator():
    """
    Verifico output del discriminatore sul sample generato random
    """
    noise = tf.random.normal([1, NOISE_DIM])
    generated_image = generator(noise, training=False)
    decision = discriminator(generated_image)
    print(decision)

#-------------------------------------------------------------------------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#-------------------------------------------------------------------------------

def discriminator_loss(real_output, fake_output):
    """
    Definisce la loss del discrimatore: tengo conto dei fail sugli esempi fake(generatore)
    e dei successi sugli esempi veri
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#-------------------------------------------------------------------------------

def generator_loss(fake_output):
    """
    Definisce la loss del generatore: tengo conto dei successi sugli esempi fake(generatore)
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#-------------------------------------------------------------------------------

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#-------------------------------------------------------------------------------

# tf.function annotation causes the function
# to be "compiled" as part of the training
@tf.function
def train_step(images):
    """
    1 - Create a random noise to feed it into the model for the image generation ;
    2 - Generate images and calculate loss values ;
    3 - Calculate gradients using loss values and model variables;
    4 - Process Gradients and Run the Optimizer ;
    """

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    # GradientTape method records operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # "gradient" method computes the gradient using
    # operations recorded in context of this tape (gen_tape and disc_tape).

    # It accepts a target (e.g., gen_loss) variable and
    # a source variable (e.g.,generator.trainable_variables)
    # target --> a list or nested structure of Tensors or Variables to be differentiated.
    # source --> a list or nested structure of Tensors or Variables.
    # target will be differentiated against elements in sources.

    # "gradient" method returns a list or nested structure of Tensors
    # (or IndexedSlices, or None), one for each element in sources.
    # Returned structure is the same as the structure of sources.
    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)


    # "apply_gradients" method processes aggregated gradients.
    # ex: optimizer.apply_gradients(zip(grads, vars))

    #Example use of apply_gradients:
    #grads = tape.gradient(loss, vars)
    #grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    # Processing aggregated gradients.
    #optimizer.apply_gradients(zip(grads, vars), experimental_aggregate_gradients=False)

    generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                            discriminator.trainable_variables))

#-------------------------------------------------------------------------------
# creo cartella in cui salvo i risultati del train come checkpoint. Inoltro genero
# un seed random costante, che riutilizzo dopo

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 100
num_examples_to_generate = 5
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

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

def train(dataset, epochs):
    """
    Definisce la funzione vera e propria di train della rete GAN :
    For each epoch:
    -) For each batch of the epoch, run the custom "train_step" function;
    -) Produce images;
    -) Save the model every 5 epochs as a checkpoint;
    -) Print out the completed epoch no. and the time spent;
    Then enerate a final image after the training is completed.
    """
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print (f'Time for epoch {epoch + 1} is {time.time()-start} sec')

    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)

#-------------------------------------------------------------------------------
if __name__=="__main__":
    train(train_dataset, EPOCHS)
