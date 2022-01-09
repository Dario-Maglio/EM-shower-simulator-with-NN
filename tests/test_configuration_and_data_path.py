"""Test configuration and dataset files exist."""

import os
import unittest
from pathlib import Path

from EM_shower_simulator import data_path



class TestCore(unittest.TestCase):
    """Test methods class for configuration and dataset."""

    def assertIsFile(self, path):
        test_path = Path(path).resolve()
        if not(test_path.is_file()):
            raise AssertionError("File does not exist: %s" % str(test_path))

    def test_dataset_path(self):
        """Test dataset path."""
        self.assertIsFile(data_path)

    def test_model_checkpoint_path(self):
        """Test model checkpoints path for the weights upload."""
        path=os.path.join('EM_shower_simulator','training_checkpoints','checkpoint')
        self.assertIsFile(Path(path))

if __name__ == "__main__":
    unittest.main()
