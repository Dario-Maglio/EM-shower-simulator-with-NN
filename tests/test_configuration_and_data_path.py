"""Test configuration and dataset files exist."""

import os
import unittest

from em_shower_simulator import GEOMETRY, default_list, CHECKP
from em_shower_simulator import N_PID, NOISE_DIM, ENERGY_NORM, ENERGY_SCALE

from em_shower_simulator.class_GAN import N_PID as GAN_PID
from em_shower_simulator.class_GAN import NOISE_DIM as GAN_DIM
from em_shower_simulator.class_GAN import ENERGY_NORM as GAN_NORM
from em_shower_simulator.class_GAN import ENERGY_SCALE as GAN_SCALE

from em_shower_simulator.make_models import N_PID as MOD_PID
from em_shower_simulator.make_models import NOISE_DIM as MOD_DIM
from em_shower_simulator.make_models import ENERGY_NORM as MOD_NORM
from em_shower_simulator.make_models import ENERGY_SCALE as MOD_SCALE

class TestCore(unittest.TestCase):
    """Test methods class for configuration and dataset."""

    def assertIsFile(self, path):
        if not os.path.isfile(path):
            raise AssertionError("File does not exist: %s" % str(path))

    def test_dataset_path(self):
        """Test dataset path."""
        for path in default_list:
            self.assertIsFile(path)

    def test_model_checkpoint_path(self):
        """Test model checkpoints path for the weights upload."""
        self.assertIsFile(CHECKP)

    def test_costants(self):
        """Assert constant values in models structure are all the same."""
        self.assertEqual(N_PID, GAN_PID)
        self.assertEqual(NOISE_DIM, GAN_DIM)
        self.assertEqual(ENERGY_NORM, GAN_NORM)
        self.assertEqual(ENERGY_SCALE, GAN_SCALE)

        self.assertEqual(N_PID, MOD_PID)
        self.assertEqual(NOISE_DIM, MOD_DIM)
        self.assertEqual(ENERGY_NORM, MOD_NORM)
        self.assertEqual(ENERGY_SCALE, MOD_SCALE)

if __name__ == "__main__":
    unittest.main()
