"""Test EM_shower_simulation module works correctly."""

import unittest

import EM_shower_simulator as EM

class TestCore(unittest.TestCase):
    """Test methods class for the module's execution."""

    def test_simulate(self):
        """Test that simulate_shower works correctly on default data."""
        self.assertEqual(EM.simulate(),0)

    def test_debug_train(self):
        """Test that the debug subroutins run without exceptions."""
        train_data = EM.debug_data_pull(EM.data_path, num_examples=5)
        train_images = train_data[0]
        EM.debug_shower(train_images)
        EM.debug_generator()
        EM.debug_discriminator(train_data)
        train_images = train_images[0, :, :, :, :]
        self.assertEqual(train_images.shape , EM.GEOMETRY)

if __name__ == "__main__":
    unittest.main()
