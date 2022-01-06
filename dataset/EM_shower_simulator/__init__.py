from . import _version
from .shower_simulator import simulate_shower as simulate
from .train_GAN import GEOMETRY, data_path, test_noise
from .train_GAN import debug_shower
from .train_GAN import debug_data_pull
from .train_GAN import debug_generator
from .train_GAN import debug_discriminator

__version__ = _version.get_versions()['version']
