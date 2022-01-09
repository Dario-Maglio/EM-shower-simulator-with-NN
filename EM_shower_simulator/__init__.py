from . import _version
from .shower_simulator import simulate_shower as simulate
from .dataset import GEOMETRY, data_path, debug_data_pull, debug_shower
from .make_models import num_examples, debug_generator, debug_discriminator
#from .train_GAN import


__version__ = _version.get_versions()['version']
