"""Primo test per documentazione"""

import unittest


class TestCore(unittest.TestCase):
    """Test methods class."""

    def test_function(self):
        """Test data format for example."""
        self.assertAlmostEqual(2.,2.)

if __name__ == "__main__":
    unittest.main()
