"""Educational LP solvers with shared scaling and independent KKT checks.

Primal simplex and predictor-corrector IPM use SciPy linear algebra only, not
SciPy optimization. PDHG implements its own matrix-vector updates. These are
teaching implementations, not substitutes for production HiGHS or PDLP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import warnings

import numpy as np
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve, LinAlgWarning
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, diags, eye

from minimal_model import Instance, build_lp, solve_greedy, validate_solution


@dataclass
class Problem:
    instance: Instance
    A: csr_matrix
    b: np.ndarray
    q: np.ndarray
    A_original: csr_matrix
    b_original: np.ndarray
    q_original: np.ndarray
    column_scale: np.ndarray
    row_scale: np.ndarray
    cost_scale: float
    edges: list
    basis: np.ndarray

    def original(self, z, y):
        return self.column_scale * z, self.cost_scale * self.row_scale * y

    def allocation(self, z):
        raw = self.column_scale * z
        x = np.zeros_like(self.instance.cost)
        for k, (i, j) in enumerate(self.edges):
            x[i, j] = raw[k]
        return x, raw[len(self.edges):len(self.edges) + len(self.instance.demand)]

    def metrics(self, z, y):
        raw, dual = self.original(z, y)
        residual = self.A_original @ raw - self.b_original
        rp = np.max(np.abs(residual)) / (1 + np.max(np.abs(self.b_original)))
        rd = np.max(np.maximum(self.A_original.T @ dual - self.q_original, 0)) / (1 + np.max(np.abs(self.q_original)))
        primal = float(self.q_original @ raw)
        dual_obj = float(self.b_original @ dual)
        gap = abs(primal - dual_obj) / (1 + abs(primal) + abs(dual_obj))
        nonneg = max(0.0, -float(raw.min())) / (1 + np.max(np.abs(raw)))
        # Extra row-wise check prevents large capacity RHS values from hiding
        # violations of a small vehicle-demand equality.
        row_residual = float(np.max(np.abs(residual) / (1 + np.abs(self.b_original))))
        score = float(max(rp, rd, gap, nonneg, row_residual))
        x, u = self.allocation(z)
        audit = validate_solution(self.instance, x, u)
        return {"objective": primal, "dual_objective": dual_obj,
                "primal_residual": float(rp), "dual_residual": float(rd),
                "duality_gap": float(gap), "nonnegative_violation": float(nonneg),
                "row_relative_residual": row_residual, "certificate_score": score,
                "absolute_business_violation": max(audit["residuals"].values()),
                "satisfaction": audit["demand_satisfaction_rate"]}


def make_problem(instance: Instance) -> Problem:
    lp = build_lp(instance)
    n, m = len(instance.demand), len(instance.compute_capacity)
    e = len(lp["edges"])
    A0 = bmat([[lp["A_eq"], csr_matrix((n, 2 * m))],
               [lp["A_ub"], eye(2 * m, format="csr")]], format="csr")
    b0 = np.concatenate((lp["b_eq"], lp["b_ub"]))
    q0 = np.concatenate((lp["c"], np.zeros(2 * m)))
    # Variable units become fractions of their individual natural upper bound.
    column = np.array([instance.demand[i] for i, _ in lp["edges"]]
                      + instance.demand.tolist() + lp["b_ub"].tolist())
    column = np.maximum(column, 1.0)
    row = 1 / np.maximum(np.abs(b0), 1.0)
    cost = float(max(1.0, np.max(np.abs(q0 * column))))
    return Problem(instance, (diags(row) @ A0 @ diags(column)).tocsr(), row * b0,
                   q0 * column / cost, A0, b0, q0, column, row, cost, lp["edges"],
                   np.arange(e, e + n + 2 * m))


@dataclass
class Solution:
    method: str
    status: str
    z: np.ndarray
    y: np.ndarray
    iterations: int
    seconds: float
    history: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


def _record(problem, z, y, history, k, start):
    metrics = problem.metrics(z, y)
    history.append({"iteration": int(k), "seconds": perf_counter() - start, **metrics})
    return metrics["certificate_score"]


def primal_simplex(problem, tolerance=1e-5, time_limit=8.0, max_iterations=10000):
    """Re-factorized primal simplex, Dantzig entering rule and Bland tie breaks."""
    start = perf_counter()
    A, b, q = problem.A.toarray(), problem.b, problem.q
    rows, cols = A.shape
    basis = problem.basis.copy()
    history, visited = [], set()
    bland = False
    status = "iteration_limit"
    z, y = np.zeros(cols), np.zeros(rows)
    for k in range(max_iterations + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            factors = lu_factor(A[:, basis], check_finite=False)
            basic = lu_solve(factors, b, check_finite=False)
            y = lu_solve(factors, q[basis], trans=1, check_finite=False)
        if float(basic.min()) < -1e-8:
            status = "numerical_failure"
            break
        z = np.zeros(cols)
        z[basis] = np.maximum(basic, 0)
        score = _record(problem, z, y, history, k, start)
        if score <= tolerance:
            status = "converged"
            break
        if perf_counter() - start >= time_limit:
            status = "time_limit"
            break
        if k == max_iterations:
            break
        reduced = q - A.T @ y
        reduced[basis] = 0.0
        candidates = np.flatnonzero(reduced < -1e-12)
        if not len(candidates):
            status = "numerical_failure"
            break
        signature = tuple(basis)
        if signature in visited:
            bland = True
        visited.add(signature)
        entering = int(candidates[0] if bland else candidates[np.argmin(reduced[candidates])])
        direction = lu_solve(factors, A[:, entering], check_finite=False)
        eligible = np.flatnonzero(direction > 1e-12)
        if not len(eligible):
            status = "unexpected_unbounded"
            break
        ratios = np.maximum(basic[eligible], 0) / direction[eligible]
        minimum = float(ratios.min())
        ties = eligible[np.abs(ratios - minimum) <= 1e-10 * (1 + abs(minimum))]
        leaving = int(ties[np.argmin(basis[ties])])
        basis[leaving] = entering
    return Solution("simplex", status, z, y, k, perf_counter() - start, history,
                    {"entering_rule": "Dantzig; Bland after repeated basis", "bland_activated": bland})


def _positive_step(value, direction):
    negative = direction < 0
    return min(1.0, float(np.min(-value[negative] / direction[negative]))) if np.any(negative) else 1.0


def interior_point(problem, tolerance=1e-5, time_limit=8.0, max_iterations=150):
    """Infeasible-start Mehrotra predictor-corrector primal-dual IPM."""
    start = perf_counter()
    A, b, q = problem.A.toarray(), problem.b, problem.q
    rows, cols = A.shape
    z, s, y = np.ones(cols), np.ones(cols), np.zeros(rows)
    history = []
    regularizations = 0
    status = "iteration_limit"
    for k in range(max_iterations + 1):
        score = _record(problem, z, y, history, k, start)
        if score <= tolerance:
            status = "converged"
            break
        if perf_counter() - start >= time_limit:
            status = "time_limit"
            break
        if k == max_iterations:
            break
        rp, rd = b - A @ z, q - A.T @ y - s
        mu = float(z @ s / cols)
        diagonal = z / s
        schur = (A * diagonal) @ A.T
        factors = None
        scale = max(1.0, float(np.max(np.diag(schur))))
        for regularization in [0.0, 1e-13, 1e-11, 1e-9, 1e-7]:
            try:
                matrix = schur if regularization == 0 else schur + np.eye(rows) * regularization * scale
                factors = cho_factor(matrix, check_finite=False)
                regularizations += regularization > 0
                break
            except np.linalg.LinAlgError:
                continue
        if factors is None:
            status = "numerical_failure"
            break

        def direction(rc):
            w = (rc - z * rd) / s
            dy = cho_solve(factors, rp - A @ w, check_finite=False)
            ds = rd - A.T @ dy
            dz = (rc - z * ds) / s
            return dz, dy, ds

        dz_aff, dy_aff, ds_aff = direction(-z * s)
        ap_aff = _positive_step(z, dz_aff)
        ad_aff = _positive_step(s, ds_aff)
        mu_aff = float((z + ap_aff * dz_aff) @ (s + ad_aff * ds_aff) / cols)
        sigma = float(np.clip((mu_aff / mu) ** 3, 0, 1))
        dz, dy, ds = direction(sigma * mu - z * s - dz_aff * ds_aff)
        ap = min(1.0, 0.995 * _positive_step(z, dz))
        ad = min(1.0, 0.995 * _positive_step(s, ds))
        z, y, s = z + ap * dz, y + ad * dy, s + ad * ds
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(y)) and np.all(np.isfinite(s))
                and np.all(z > 0) and np.all(s > 0)):
            status = "numerical_failure"
            break
    return Solution("ipm", status, z, y, k, perf_counter() - start, history,
                    {"centering": "Mehrotra predictor-corrector", "regularized_factorizations": regularizations})


def pdhg(problem, tolerance=1e-5, time_limit=8.0, max_iterations=100000, check_every=50):
    """PDHG with fixed diagonal step sizes and no hidden optimization calls.

T_j = 0.9 / sum_i |A_ij| and S_i = 0.9 / sum_j |A_ij| give a
preconditioned operator norm below one. Both latest and ergodic iterates are
checked; the better certificate is returned. This is not full PDLP.
"""
    start = perf_counter()
    A, b, q = problem.A, problem.b, problem.q
    AT = A.T.tocsr()
    abs_a = abs(A)
    tau = 0.9 / np.maximum(np.asarray(abs_a.sum(axis=0)).ravel(), 1e-12)
    sigma = 0.9 / np.maximum(np.asarray(abs_a.sum(axis=1)).ravel(), 1e-12)
    z = np.zeros(A.shape[1])
    # The known all-unserved feasible point is available without an oracle.
    z[problem.basis] = problem.b / A[:, problem.basis].diagonal()
    v = np.zeros(A.shape[0])
    extrapolated = z.copy()
    avg_z, avg_v = np.zeros_like(z), np.zeros_like(v)
    history = []
    initial = _record(problem, z, -v, history, 0, start)
    best_score, best_z, best_y = initial, z.copy(), -v.copy()
    status, selected = "iteration_limit", "latest"
    if initial <= tolerance:
        return Solution("pdhg", "converged", z, -v, 0, perf_counter() - start, history)
    for k in range(1, max_iterations + 1):
        v = v + sigma * (A @ extrapolated - b)
        new_z = np.maximum(z - tau * (q + AT @ v), 0)
        extrapolated = 2 * new_z - z
        z = new_z
        avg_z += (z - avg_z) / k
        avg_v += (v - avg_v) / k
        if k % check_every and k != max_iterations:
            continue
        latest = problem.metrics(z, -v)
        average = problem.metrics(avg_z, -avg_v)
        if latest["certificate_score"] <= average["certificate_score"]:
            candidate, zz, yy, choice = latest, z, -v, "latest"
        else:
            candidate, zz, yy, choice = average, avg_z, -avg_v, "ergodic_average"
        history.append({"iteration": k, "seconds": perf_counter() - start,
                        "candidate": choice, **candidate})
        if candidate["certificate_score"] < best_score:
            best_score, best_z, best_y, selected = candidate["certificate_score"], zz.copy(), yy.copy(), choice
        if best_score <= tolerance:
            status = "converged"
            break
        if perf_counter() - start >= time_limit:
            status = "time_limit"
            break
    return Solution("pdhg", status, best_z, best_y, k, perf_counter() - start, history,
                    {"step_sizes": "fixed diagonal preconditioner, safety=0.9", "returned_iterate": selected})


def highs(problem, tolerance=1e-5, time_limit=8.0, **unused):
    start = perf_counter()
    r = linprog(problem.q, A_eq=problem.A, b_eq=problem.b, bounds=(0, None),
                method="highs-ds", options={"time_limit": time_limit,
                "primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    if not r.success:
        raise RuntimeError(r.message)
    history = []
    score = _record(problem, r.x, r.eqlin.marginals, history, int(r.nit), start)
    return Solution("highs", "converged" if score <= tolerance else "certificate_failed",
                    r.x, r.eqlin.marginals, int(r.nit), perf_counter() - start, history,
                    {"presolve": True, "solver_message": r.message})


def greedy(problem, tolerance=1e-5, **unused):
    start = perf_counter()
    result = solve_greedy(problem.instance)
    x, u = np.array(result["allocation"]), np.array(result["unserved"])
    raw = np.array([x[i, j] for i, j in problem.edges] + u.tolist()
                   + (problem.instance.compute_capacity - np.array(result["verification"]["compute_usage"])).tolist()
                   + (problem.instance.upload_capacity - np.array(result["verification"]["upload_usage"])).tolist())
    z, y = raw / problem.column_scale, np.zeros(problem.A.shape[0])
    history = []
    _record(problem, z, y, history, 0, start)
    return Solution("greedy", "feasible_baseline", z, y, 0, perf_counter() - start, history,
                    {"certificate_available": False})


SOLVERS = {"simplex": primal_simplex, "ipm": interior_point, "pdhg": pdhg,
           "highs": highs, "greedy": greedy}
