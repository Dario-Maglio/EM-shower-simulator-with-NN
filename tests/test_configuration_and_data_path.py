"""Test configuration and dataset files exist."""

import os
import unittest
from pathlib import Path



class TestCore(unittest.TestCase):
    """Test methods class for configuration and dataset."""

    def assertIsFile(self, path):
        tpath = Path(path).resolve()
        if not(tpath.is_file()):
            raise AssertionError("File does not exist: %s" % str(tpath))

    def test_dataset_path(self):
        """Test dataset path."""
        path=Path(os.path.join('dataset','filtered_data','data_MVA.root'))
        self.assertIsFile(path)

    def test_model_checkpoint_path(self):
        """Test model checkpoints path for the weights upload."""
        path=os.path.join('EM_shower_simulator','training_checkpoints','checkpoint')
        self.assertIsFile(Path(path))

if __name__ == "__main__":
    unittest.main()
