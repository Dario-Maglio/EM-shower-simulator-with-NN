"""Test configuration and dataset files exist."""

import os
import unittest

from EM_shower_simulator import default_list, CHECKP

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
        #self.assertEqual(NOISE_DIM , NOISE_DIM) among different files
        pass

if __name__ == "__main__":
    unittest.main()
