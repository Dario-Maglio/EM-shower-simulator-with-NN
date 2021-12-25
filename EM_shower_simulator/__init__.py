"""Package information"""

from . import _version
__version__ = _version.get_versions()['version']

from .shower_simulator import simulate_shower

PACKAGE_NAME = 'EM_shower_simulator'
AUTHOR = 'Dario Cafasso, Daniele Passaro'
AUTHOR_EMAIL = 'cafasso.dario@gmail.com, passaro.?'
DESCRIPTION = '....description......'
URL = '....url.....'
