"""Analytical and boundary-case checks for the model and independent validator."""
import json
from pathlib import Path
import unittest

import numpy as np

from minimal_model import Instance, solve_greedy, solve_reference, validate_solution

DATA = Path(__file__).resolve().parents[1] / "data/two_vehicles.json"


class MinimalModelTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(DATA.read_text(encoding="utf-8"))

    def test_analytical_optimum_and_greedy_counterexample(self):
        instance = Instance.from_dict(self.data)
        result = solve_reference(instance)
        self.assertAlmostEqual(result["verification"]["objective"], 15.5)
        np.testing.assert_allclose(result["allocation"], [[0, 5], [5, 0]])
        self.assertAlmostEqual(solve_greedy(instance)["verification"]["objective"], 55.0)

    def test_zero_compute_capacity_leaves_everything_unserved(self):
        self.data["compute_capacity"] = [0, 0]
        result = solve_reference(Instance.from_dict(self.data))
        np.testing.assert_allclose(result["unserved"], [5, 5])
        self.assertAlmostEqual(result["verification"]["objective"], 200.0)

    def test_disconnected_vehicle_cannot_be_allocated(self):
        self.data["cost"] = [[None, None], [1.1, 10]]
        result = solve_reference(Instance.from_dict(self.data))
        self.assertAlmostEqual(result["unserved"][0], 5.0)
        np.testing.assert_allclose(result["allocation"][0], [0, 0])
        self.assertAlmostEqual(result["verification"]["objective"], 105.5)

    def test_upload_bottleneck_has_known_optimum(self):
        data = {"name": "upload_bottleneck", "demand": [5], "compute_per_workload": [1],
                "upload_per_workload": [2], "compute_capacity": [10], "upload_capacity": [4],
                "cost": [[1]], "unserved_penalty": [20]}
        result = solve_reference(Instance.from_dict(data))
        np.testing.assert_allclose(result["allocation"], [[2]])
        np.testing.assert_allclose(result["unserved"], [3])
        self.assertAlmostEqual(result["verification"]["objective"], 62.0)

    def test_independent_validator_rejects_capacity_violation(self):
        instance = Instance.from_dict(self.data)
        checks = validate_solution(instance, np.array([[5.0, 0], [5.0, 0]]), np.zeros(2))
        self.assertFalse(checks["passed"])
        self.assertAlmostEqual(checks["residuals"]["demand_equality_max_abs"], 0.0)
        self.assertAlmostEqual(checks["residuals"]["compute_capacity_max_violation"], 5.0)

    def test_negative_capacity_is_rejected(self):
        self.data["compute_capacity"][0] = -1
        with self.assertRaises(ValueError):
            Instance.from_dict(self.data)


if __name__ == "__main__":
    unittest.main()
