"""Compatibility test entry for the current JSON/assignment network workflow."""
import sys
import unittest

sys.path.insert(0, "/home/lxl/dataTopology/20260915final/tests")
sys.path.insert(0, "/home/lxl/dataTopology/20260915final")
from test_assignment_network import AssignmentTests, AddressAndNetworkTests, JsonTests, RRFunctionTests

if __name__ == "__main__":
    unittest.main()
