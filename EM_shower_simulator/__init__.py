import os
from . import _version
from .shower_simulator import simulate_shower as simulate

from .dataset import debug_data_pull, debug_shower
from .make_models import debug_generator, debug_discriminator
from .class_GAN import test_noise

from .constants import GEOMETRY, FILE_LIST, default_list, CHECKP

__version__ = _version.get_versions()['version']
