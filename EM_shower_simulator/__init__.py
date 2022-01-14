import os
from . import _version
from .shower_simulator import simulate_shower as simulate

from .dataset import GEOMETRY, data_path, debug_data_pull, debug_shower
from .make_models import debug_generator, debug_discriminator
from .class_GAN import test_noise
check_path = os.path.join('EM_shower_simulator','training_checkpoints','checkpoint')

__version__ = _version.get_versions()['version']
