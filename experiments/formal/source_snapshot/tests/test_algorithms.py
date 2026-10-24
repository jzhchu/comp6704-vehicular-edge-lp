import json
from pathlib import Path
import unittest
import numpy as np
from algorithms import make_problem, SOLVERS
from generate_instances import generate
from minimal_model import Instance, validate_solution


class SolverTests(unittest.TestCase):
    def check_methods(self, data):
        instance = Instance.from_dict(data)
        p = make_problem(instance)
        reference = SOLVERS["highs"](p)
        reference_cost = p.metrics(reference.z, reference.y)["objective"]
        for name in ["simplex", "ipm", "pdhg"]:
            with self.subTest(method=name, instance=instance.name):
                solution = SOLVERS[name](p, tolerance=1e-5, time_limit=10)
                metrics = p.metrics(solution.z, solution.y)
                self.assertEqual(solution.status, "converged", metrics)
                self.assertLessEqual(metrics["certificate_score"], 1e-5)
                self.assertLess(abs(metrics["objective"] - reference_cost) / max(1, abs(reference_cost)), 5e-4)
                x, u = p.allocation(solution.z)
                self.assertLessEqual(max(validate_solution(instance, x, u)["residuals"].values()), 1e-3)

    def test_analytical_example(self):
        self.check_methods(json.loads((Path(__file__).resolve().parents[1] / "data/two_vehicles.json").read_text()))

    def test_zero_resources_and_disconnected_vehicles(self):
        self.check_methods({"name": "zero_resources", "demand": [2, 3],
            "compute_per_workload": [1, 2], "upload_per_workload": [1, 1],
            "compute_capacity": [0], "upload_capacity": [0],
            "cost": [[1], [None]], "unserved_penalty": [20, 20]})

    def test_seeded_instances_against_reference(self):
        for seed in [7, 13, 29]:
            _, data = generate(n=8, m=3, rho_compute=0.8, rho_upload=1.1, radius=0.4, seed=seed)
            self.check_methods(data)

    def test_zero_demand(self):
        self.check_methods({"name": "zero_demand", "demand": [0], "compute_per_workload": [1],
            "upload_per_workload": [1], "compute_capacity": [0], "upload_capacity": [0],
            "cost": [[None]], "unserved_penalty": [20]})

    def test_scaling_round_trip(self):
        instance, _ = generate(n=10, seed=3)
        p = make_problem(instance)
        z = np.ones(p.A.shape[1]); y = np.ones(p.A.shape[0])
        raw, dual = p.original(z, y)
        np.testing.assert_allclose(p.A @ z - p.b, p.row_scale * (p.A_original @ raw - p.b_original))
        self.assertAlmostEqual(p.q @ z * p.cost_scale, p.q_original @ raw)
        self.assertAlmostEqual(p.b @ y * p.cost_scale, p.b_original @ dual)


if __name__ == "__main__":
    unittest.main()
