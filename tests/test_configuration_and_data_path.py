"""Tests that the configuration and dataset files exists."""

import unittest
import pathlib as pl

from EM_shower_simulator.config import n_features, DEFAULT_FEATURES

class TestCore(unittest.TestCase):
    """Test methods class for configuration and dataset."""

    def test_config(self):
        """Test default data format."""
        self.assertEqual(len(DEFAULT_FEATURES), n_features)

    def assertIsFile(self, path):
        tpath = pl.Path(path).resolve()
        if not(tpath.is_file()):
            raise AssertionError("File does not exist: %s" % str(tpath))

    def test_dataset_path(self):
        """Test dataset path."""
        path=pl.Path('Dataset/Filtered_data/data_MVA.root')
        self.assertIsFile(path)

if __name__ == "__main__":
    unittest.main()
