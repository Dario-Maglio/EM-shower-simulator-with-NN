"""Tests that the EM_shower_simulation module works correctly on default data."""

import unittest

from EM_shower_simulator.config import DEFAULT_FEATURES
import EM_shower_simulator.EM_shower_simulator as EM


class TestCore(unittest.TestCase):
    """Test methods class for the module's execution."""

    def test_simulate(self):
        """Test simulate_shower works correctly on default data."""
        self.assertEqual(EM.simulate_shower(DEFAULT_FEATURES),0)

if __name__ == "__main__":
    unittest.main()
