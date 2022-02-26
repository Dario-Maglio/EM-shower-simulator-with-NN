Network structure
-----------------

Generator model scheme:

  .. image:: https://raw.githubusercontent.com/Dario-Caf/EM-shower-simulator-with-NN/main/EM_shower_simulator/model_plot/cgan-generator.png
    :width: 800
    :alt: Alternative text

Discriminator model scheme:

  .. image:: https://raw.githubusercontent.com/Dario-Caf/EM-shower-simulator-with-NN/main/EM_shower_simulator/model_plot/cgan-discriminator.png
    :width: 800
    :alt: Alternative text

To randomly simulate the EM showers we decided to build
a GAN (generative adversarial network) with features imple-
mented ad hoc for our purpose. We used Keras layers and the
TensorFlow package to build and manage our network.
A GAN consists of a generator network and a discriminator
one, which are trained against each other: the generator is
awarded if it manages to cheat the discriminator, while the lat-
ter is awarded if it succeeds to recognize fake samples coming
from the generator.
The generator network has 3 inputs: normally distributed random noise from a
latent space R^1024, integer primary particle ID (Pin = 0 for electrons, 1 for
photons and 2 for positrons) and primary particle energy (Ein, from 1.0 to
30.0). Primary particle ID is connected to an embedding layer to create a
discrete category for each particle, and then it is concatenated with the other
inputs. A succession of Conv3DTranspose, BatchNormalization
and LeakyRelu layers is present. Its final output (hereon said
G) is a (None, 12, 25, 25, 1) tensor whose activation function is
tanh, so that each pixel takes value in [−1; 1]. Conv3DTranspose
layers deconvolve inputs maintaining high levels of connectivity
between feature maps, BatchNormalization is important to help
with weight initialization and gradient stability, and LeakyRelu
activation is useful to prevent vanishing gradient and to provide
a higher level of sparsity.
The discriminator network takes as only input a
(None, 12, 25, 25, 1) tensor, which is the images vector from
the generator or one from the Geant4 dataset. It works as a
classifier: it decides (outputs a decision ”D”) whether the input
is fake (G) or real (I). Moreover, it decides the primary particle
generating the shower (Plabel GAN/GEANT) and has to recognize the
label primary particle energy (EGAN/GEANT) (actually, this label helps
to relate shower’s shape to primary particle energy). For this
purpose, the input tensor is concatenated with a computed
tensor whose elements are the non-normalized energies de-
posited in each detector’s layer. This operation is performed
with a costume Lambda layer. In this way, we observed that
the discriminator better recognizes fake from original showers,
enhancing the generator to produce images with the ”correct”
energy deposition per layer. Then a succession of Conv3D,
Pooling3D and LeayRelu layers is applied to the input tensor.
Pooling layers help to resolve spatial characteristics of the
shower, like their maximum and shape. After a convolutive
stage, a Flatten layer that passes the output tensor to 3 different
paths is present. Each one represents one of the decisions
above. Dense layers are present to increase the deepness of
the network and increase connection among neurons. To avoid
”mode collapse”, which is an often common evolution of GANs,
we inserted between convolutive layers a costume Lambda
layer that performs ”minibatch standard deviation discrimina-
tion”. This technique has been conceived by NVIDIA researchers
while developing a GAN to create ultra-realistic deep-fake faces.
Its role is to compute the standard deviation of a group of input
features. It concatenates the result back to the input tensor
and passes the resulting tensor to a convolutive layer. In this
way, the discriminator should be able to distinguish fake from
original samples by looking at the standard deviation among
features: for a diversified real dataset it will be high, while
for a fake dataset where the generator has undergone mode
collapse it will be small.Recognizing it, the discriminator would enhance the
generator to produce diversified samples. We adapted this idea to our
case computing the standard deviation for each image (layer of
the calorimeter) instead of the whole images vector, thus
introducing a ”sparsity” metrics for each layer of the detector.
