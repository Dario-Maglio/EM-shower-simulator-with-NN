"""Test EM_shower_simulation module works correctly."""

import unittest

import em_shower_simulator as em

class TestCore(unittest.TestCase):
    """Test methods class for the module's execution."""

    def test_simulate(self):
        """Test that simulate_shower works correctly on default data."""
        self.assertEqual(em.simulate(),0)
        pass

    def test_debug_models(self):
        """Test that the debug subroutins run without exceptions."""
        train_data = em.debug_data_pull(em.default_list)
        train_images = train_data[0]
        em.debug_shower(train_images)
        em.debug_generator(em.test_noise)
        em.debug_discriminator(train_images)
        train_images = train_images[0, :, :, :, :]
        self.assertEqual(train_images.shape , em.GEOMETRY)

if __name__ == "__main__":
    unittest.main()
