"""Tests that the EM_shower_simulation module works correctly on default data."""

import unittest

from EM_shower_simulator.shower_simulator import simulate_shower as simulate


class TestCore(unittest.TestCase):
    """Test methods class for the module's execution."""

    def test_simulate(self):
        """Test simulate_shower works correctly on default data."""
        self.assertEqual(simulate(),0)

if __name__ == "__main__":
    unittest.main()
