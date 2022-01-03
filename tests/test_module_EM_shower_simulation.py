"""Test EM_shower_simulation module works correctly."""

import unittest

import EM_shower_simulator as EM



class TestCore(unittest.TestCase):
    """Test methods class for the module's execution."""

    def test_simulate(self):
        """Test that simulate_shower works correctly on default data."""
        self.assertEqual(EM.simulate(),0)

if __name__ == "__main__":
    unittest.main()
