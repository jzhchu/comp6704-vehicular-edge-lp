"""Seeded synthetic one-window instances, matching the proposal's units."""
import numpy as np
from minimal_model import Instance


def generate(n=30, m=4, rho_compute=1.0, rho_upload=1.0, radius=0.25, seed=11):
    rng = np.random.default_rng(seed)
    demand = rng.uniform(1, 5, n)
    a, b = rng.uniform(0.5, 2, n), rng.uniform(0.2, 1, n)
    position, rsus = rng.uniform(0, 1, n), np.linspace(0, 1, m)
    distance = abs(position[:, None] - rsus[None, :])
    costs = 0.5 + 2 * distance + 0.5 * rng.uniform(0, 1, m)[None, :]
    compute_weights, upload_weights = rng.uniform(0.5, 1.5, m), rng.uniform(0.5, 1.5, m)
    compute = rho_compute * (a @ demand) * compute_weights / compute_weights.sum()
    upload = rho_upload * (b @ demand) * upload_weights / upload_weights.sum()
    data = {"name": f"n{n}_m{m}_c{rho_compute:g}_b{rho_upload:g}_r{radius:g}_s{seed}",
            "demand": demand.tolist(), "compute_per_workload": a.tolist(),
            "upload_per_workload": b.tolist(), "compute_capacity": compute.tolist(),
            "upload_capacity": upload.tolist(), "unserved_penalty": [20.0] * n,
            "cost": [[float(costs[i, j]) if distance[i, j] <= radius else None for j in range(m)] for i in range(n)]}
    return Instance.from_dict(data), data
