"""Reference LP for divisible vehicular workloads, solved by SciPy/HiGHS.

This is a modeling and correctness experiment, not a custom LP solver or a
multi-algorithm performance study. A deterministic greedy baseline illustrates
the effect of competing for limited RSU capacity.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Instance:
    name: str
    demand: np.ndarray
    compute_per_workload: np.ndarray
    upload_per_workload: np.ndarray
    compute_capacity: np.ndarray
    upload_capacity: np.ndarray
    cost: np.ndarray
    unserved_penalty: np.ndarray

    @classmethod
    def from_dict(cls, data: dict) -> "Instance":
        def vector(key: str) -> np.ndarray:
            v = np.asarray(data[key], dtype=float)
            if v.ndim != 1 or not np.all(np.isfinite(v)):
                raise ValueError(f"{key} must be a finite one-dimensional vector")
            return v

        d, a, b = (vector(k) for k in (
            "demand", "compute_per_workload", "upload_per_workload"
        ))
        c_cap, b_cap, penalty = (vector(k) for k in (
            "compute_capacity", "upload_capacity", "unserved_penalty"
        ))
        n, m = len(d), len(c_cap)
        if n == 0 or m == 0:
            raise ValueError("At least one vehicle and one RSU are required")
        if len(a) != n or len(b) != n or len(penalty) != n or len(b_cap) != m:
            raise ValueError("Vehicle and RSU vector lengths do not match")
        if np.any(d < 0) or np.any(c_cap < 0) or np.any(b_cap < 0):
            raise ValueError("Demand and capacities must be nonnegative")
        if np.any(a <= 0) or np.any(b <= 0):
            raise ValueError("Unit resource consumptions must be positive")
        if np.any(penalty < 0):
            raise ValueError("Unserved penalties must be nonnegative")
        # A JSON null means that a vehicle cannot connect to that RSU.
        raw_cost = data["cost"]
        if not isinstance(raw_cost, list) or len(raw_cost) != n:
            raise ValueError("cost must have one row per vehicle")
        if any(not isinstance(row, list) or len(row) != m for row in raw_cost):
            raise ValueError("cost must have one column per RSU")
        cost = np.full((n, m), np.nan)
        for i, row in enumerate(raw_cost):
            for j, value in enumerate(row):
                if value is not None:
                    cost[i, j] = float(value)
                    if not np.isfinite(cost[i, j]) or cost[i, j] < 0:
                        raise ValueError("Connected-edge costs must be finite and nonnegative")
        return cls(str(data.get("name", "unnamed")), d, a, b, c_cap, b_cap, cost, penalty)


def validate_solution(instance: Instance, x: np.ndarray, u: np.ndarray,
                      tolerance: float = 1e-8) -> dict:
    """Recompute constraints from physical parameters, without LP matrices."""
    n, m = len(instance.demand), len(instance.compute_capacity)
    if x.shape != (n, m) or u.shape != (n,):
        raise ValueError("Solution dimensions do not match the instance")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
        raise ValueError("Solution contains non-finite values")
    compute = (instance.compute_per_workload[:, None] * x).sum(axis=0)
    upload = (instance.upload_per_workload[:, None] * x).sum(axis=0)
    connected = np.isfinite(instance.cost)
    processing_cost = float(np.sum(instance.cost[connected] * x[connected]))
    penalty_cost = float(instance.unserved_penalty @ u)
    residuals = {
        "demand_equality_max_abs": float(np.max(np.abs(x.sum(axis=1) + u - instance.demand))),
        "compute_capacity_max_violation": float(np.max(np.maximum(compute - instance.compute_capacity, 0))),
        "upload_capacity_max_violation": float(np.max(np.maximum(upload - instance.upload_capacity, 0))),
        "nonnegativity_max_violation": float(max(0.0, -float(x.min()), -float(u.min()))),
        "disconnected_edge_max_abs": float(np.max(np.abs(x[~connected]))) if np.any(~connected) else 0.0,
    }
    total_demand = float(instance.demand.sum())
    return {
        "passed": all(value <= tolerance for value in residuals.values()),
        "absolute_tolerance": tolerance,
        "residuals": residuals,
        "processing_cost": processing_cost,
        "unserved_penalty_cost": penalty_cost,
        "objective": processing_cost + penalty_cost,
        "demand_satisfaction_rate": 1.0 - float(u.sum()) / total_demand if total_demand else None,
        "compute_usage": compute.tolist(),
        "upload_usage": upload.tolist(),
    }


def build_lp(instance: Instance) -> dict:
    """Variable order: connected x_ij entries, followed by every u_i."""
    n, m = len(instance.demand), len(instance.compute_capacity)
    edges = [tuple(map(int, e)) for e in np.argwhere(np.isfinite(instance.cost))]
    edge_count = len(edges)
    variable_count = edge_count + n
    objective = np.zeros(variable_count)
    eq_rows, eq_cols, eq_values = [], [], []
    ub_rows, ub_cols, ub_values = [], [], []
    for k, (i, j) in enumerate(edges):
        objective[k] = instance.cost[i, j]
        eq_rows.append(i); eq_cols.append(k); eq_values.append(1.0)
        ub_rows.extend((j, m + j)); ub_cols.extend((k, k))
        ub_values.extend((instance.compute_per_workload[i], instance.upload_per_workload[i]))
    for i in range(n):
        objective[edge_count + i] = instance.unserved_penalty[i]
        eq_rows.append(i); eq_cols.append(edge_count + i); eq_values.append(1.0)
    return {
        "edges": edges,
        "c": objective,
        "A_eq": coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(n, variable_count)).tocsr(),
        "b_eq": instance.demand.copy(),
        "A_ub": coo_matrix((ub_values, (ub_rows, ub_cols)), shape=(2 * m, variable_count)).tocsr(),
        "b_ub": np.concatenate((instance.compute_capacity, instance.upload_capacity)),
    }


def solve_reference(instance: Instance) -> dict:
    model = build_lp(instance)
    start = perf_counter()
    result = linprog(
        model["c"], A_ub=model["A_ub"], b_ub=model["b_ub"],
        A_eq=model["A_eq"], b_eq=model["b_eq"], bounds=(0, None),
        method="highs-ds",
        options={"primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9},
    )
    seconds = perf_counter() - start
    if not result.success:
        raise RuntimeError(f"HiGHS failed: {result.status}: {result.message}")
    x = np.zeros_like(instance.cost)
    for k, (i, j) in enumerate(model["edges"]):
        x[i, j] = result.x[k]
    u = result.x[len(model["edges"]):].copy()
    # Normalize the sign of exact zeros without altering small nonzero values.
    x[x == 0] = 0.0
    u[u == 0] = 0.0
    checks = validate_solution(instance, x, u)
    if not checks["passed"]:
        raise RuntimeError(f"Independent feasibility check failed: {checks['residuals']}")
    if abs(checks["objective"] - float(result.fun)) > 1e-8:
        raise RuntimeError("Recomputed objective disagrees with the solver objective")
    return {
        "method": "SciPy linprog / HiGHS dual simplex (highs-ds)",
        "status": str(result.message),
        "solver_objective": float(result.fun),
        "solver_seconds": seconds,
        "solver_iterations": int(result.nit),
        "variables": int(len(model["c"])),
        "equality_constraints": int(model["A_eq"].shape[0]),
        "capacity_constraints": int(model["A_ub"].shape[0]),
        "allocation": x.tolist(),
        "unserved": u.tolist(),
        "verification": checks,
    }


def solve_greedy(instance: Instance) -> dict:
    """Serve higher penalties first; break ties by vehicle/RSU index."""
    n, m = len(instance.demand), len(instance.compute_capacity)
    x = np.zeros((n, m)); u = instance.demand.copy()
    remaining_compute = instance.compute_capacity.copy()
    remaining_upload = instance.upload_capacity.copy()
    for i in sorted(range(n), key=lambda i: (-instance.unserved_penalty[i], i)):
        candidates = [j for j in range(m) if np.isfinite(instance.cost[i, j])]
        for j in sorted(candidates, key=lambda j: (instance.cost[i, j], j)):
            if instance.cost[i, j] >= instance.unserved_penalty[i]:
                continue
            amount = max(0.0, min(u[i], remaining_compute[j] / instance.compute_per_workload[i],
                                  remaining_upload[j] / instance.upload_per_workload[i]))
            x[i, j] += amount
            u[i] -= amount
            remaining_compute[j] -= instance.compute_per_workload[i] * amount
            remaining_upload[j] -= instance.upload_per_workload[i] * amount
    checks = validate_solution(instance, x, u)
    if not checks["passed"]:
        raise RuntimeError(f"Greedy feasibility check failed: {checks['residuals']}")
    return {"method": "Penalty-first, lowest-cost feasible greedy", "allocation": x.tolist(),
            "unserved": u.tolist(), "verification": checks}


def save_results(instance: Instance, reference: dict, greedy: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "instance_name": instance.name,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(),
                        "machine": platform.machine()},
        "reference": reference, "greedy": greedy,
        "note": "Single-run runtime is diagnostic, not a benchmark result. Input is synthetic.",
    }
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "allocation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "vehicle", "rsu", "workload"])
        for label, solution in [("reference", reference), ("greedy", greedy)]:
            for i, row in enumerate(solution["allocation"]):
                for j, value in enumerate(row):
                    if np.isfinite(instance.cost[i, j]):
                        writer.writerow([label, i + 1, j + 1, value])
            for i, value in enumerate(solution["unserved"]):
                writer.writerow([label, i + 1, "unserved", value])
    with (output / "resources.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "rsu", "compute_used", "compute_capacity", "upload_used", "upload_capacity"])
        for label, solution in [("reference", reference), ("greedy", greedy)]:
            checks = solution["verification"]
            for j in range(len(instance.compute_capacity)):
                writer.writerow([label, j + 1, checks["compute_usage"][j], instance.compute_capacity[j],
                                 checks["upload_usage"][j], instance.upload_capacity[j]])
    lines = ["车联网边缘计算最小模型运行结果", "", f"场景：{instance.name}",
             f"参考解总成本：{reference['verification']['objective']:.6f}",
             f"贪心解总成本：{greedy['verification']['objective']:.6f}",
             f"参考解需求满足率：{reference['verification']['demand_satisfaction_rate']:.2%}" if instance.demand.sum() else "参考解需求满足率：不适用（总需求为零）",
             f"参考解约束检查：{'通过' if reference['verification']['passed'] else '失败'}",
             f"贪心解约束检查：{'通过' if greedy['verification']['passed'] else '失败'}", "", "参考解分配"]
    for i, row in enumerate(reference["allocation"]):
        parts = [f"RSU {j+1} = {v:.6f}" for j, v in enumerate(row)]
        lines.append(f"车辆 {i+1}：" + "，".join(parts) + f"，未满足 = {reference['unserved'][i]:.6f}")
    lines.extend(["", "本结果来自合成小例子，用于确认模型可运行及约束正确。", "多算法比较结果另存于 experiments/batch/summary.txt，由批量实验脚本生成。"])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=BASE_DIR / "data/two_vehicles.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args()
    instance = Instance.from_dict(json.loads(args.instance.read_text(encoding="utf-8")))
    reference = solve_reference(instance)
    greedy = solve_greedy(instance)
    save_results(instance, reference, greedy, args.output)
    print((args.output / "summary.txt").read_text(encoding="utf-8"))
    print(f"结果目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()
