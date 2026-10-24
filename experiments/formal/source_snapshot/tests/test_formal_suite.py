import json
from pathlib import Path
import unittest

import numpy as np

from algorithms import SOLVERS, make_problem
from formal_suite import build_jobs, certified_reference
from generate_instances import generate
from minimal_model import Instance


class FormalSuiteTests(unittest.TestCase):
    def test_scaling_ablation_preserves_physical_optimum(self):
        instance, _ = generate(n=10, m=3, seed=53, rho_compute=.7, rho_upload=1.6)
        costs = []
        for scaling in [False, True]:
            p = make_problem(instance, scaling=scaling)
            for method in ["highs", "simplex", "ipm"]:
                s = SOLVERS[method](p, tolerance=1e-7)
                self.assertEqual(s.status, "converged")
                metric = p.metrics(s.z, s.y)
                self.assertLessEqual(metric["certificate_score"], 1e-7)
                costs.append(metric["objective"])
        np.testing.assert_allclose(costs, costs[0], rtol=1e-6)

    def test_raw_dual_predicts_known_capacity_change(self):
        data = dict(name="known_shadow", demand=[10], compute_per_workload=[2],
            upload_per_workload=[1], compute_capacity=[8], upload_capacity=[100],
            cost=[[1]], unserved_penalty=[20])
        baseline = certified_reference(Instance.from_dict(data))
        self.assertAlmostEqual(baseline["metrics"]["objective"], 124.)
        price = -baseline["raw_dual"][1]
        self.assertAlmostEqual(price, 9.5)
        data["compute_capacity"][0] += .5
        changed = certified_reference(Instance.from_dict(data))
        self.assertAlmostEqual(124.-changed["metrics"]["objective"], .5*price)

    def test_fixed_design_counts_and_capacity_pairing(self):
        config = json.loads((Path(__file__).resolve().parents[1]/"configs/formal_suite.json").read_text())
        inputs, jobs = build_jobs(config)
        self.assertEqual(len(inputs), 90)
        self.assertEqual(len(jobs), 1980)
        self.assertEqual(len({j["job_id"] for j in jobs}), 1980)
        for seed in config["seeds"]:
            _, a = generate(n=30, m=4, rho_compute=.4, rho_upload=1.6, seed=seed)
            _, b = generate(n=30, m=4, rho_compute=1.3, rho_upload=1.6, seed=seed)
            for key in ["demand", "cost", "compute_per_workload", "upload_per_workload", "upload_capacity"]:
                self.assertEqual(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
