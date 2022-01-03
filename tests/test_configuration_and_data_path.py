"""Test configuration and dataset files exist."""

import unittest
import pathlib as pl



class TestCore(unittest.TestCase):
    """Test methods class for configuration and dataset."""

    def assertIsFile(self, path):
        tpath = pl.Path(path).resolve()
        if not(tpath.is_file()):
            raise AssertionError("File does not exist: %s" % str(tpath))

    def test_dataset_path(self):
        """Test dataset path."""
        path=pl.Path('dataset/filtered_data/data_MVA.root')
        self.assertIsFile(path)

if __name__ == "__main__":
    unittest.main()
