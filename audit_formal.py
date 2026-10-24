"""Recompute certificates, physical constraints and labels from saved raw vectors."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from algorithms import make_problem
from formal_suite import BASE, build_jobs, dump, sha
from minimal_model import Instance, validate_solution


def check_payload(instance, payload):
    # Work in physical units, independent of the scaling used during the solve.
    p = make_problem(instance, scaling=False)
    raw, dual = np.array(payload["raw_primal"]), np.array(payload["raw_dual"])
    assert np.isfinite(raw).all() and np.isfinite(dual).all()
    metrics = p.metrics(raw, dual)
    x, u = p.allocation(raw)
    np.testing.assert_allclose(x, payload["allocation"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u, payload["unserved"], rtol=1e-12, atol=1e-12)
    business = validate_solution(instance, x, u)
    np.testing.assert_allclose(metrics["objective"], business["objective"], rtol=1e-12, atol=1e-10)
    for key, value in metrics.items():
        np.testing.assert_allclose(value, payload["metrics"][key], rtol=1e-10, atol=1e-11)
    # Independent business feasibility, including forbidden connections.
    violations = [np.max(abs(x.sum(axis=1)+u-instance.demand)/(1+instance.demand)),
        np.max(np.maximum(np.array(business["compute_usage"])-instance.compute_capacity, 0)/(1+instance.compute_capacity)),
        np.max(np.maximum(np.array(business["upload_usage"])-instance.upload_capacity, 0)/(1+instance.upload_capacity)),
        max(0., -float(x.min()), -float(u.min())),
        float(np.max(abs(x[np.isnan(instance.cost)]), initial=0.))]
    return metrics, float(max(violations))


def audit(output):
    manifest = json.loads((output/"manifest.json").read_text())
    config = json.loads((output/"config.json").read_text())
    assert manifest["config"] == config
    assert sha(output/"config.json") == manifest["config_sha256"]
    assert sha(output/"jobs.json") == manifest["jobs_sha256"]
    for name, expected in manifest["source_sha256"].items():
        assert sha(output/manifest["source_snapshot"]/name) == expected, name
    for name, expected in manifest.get("result_sha256", {}).items():
        assert sha(output/name) == expected, name
    inputs, expected_jobs = build_jobs(config)
    jobs = json.loads((output/"jobs.json").read_text())
    assert jobs == expected_jobs
    instances, references = {}, {}
    for name, expected in manifest["instance_sha256"].items():
        path = output/"instances"/f"{name}.json"
        assert sha(path) == expected
        data = json.loads(path.read_text())
        assert data == inputs[name]
        instance = instances[name] = Instance.from_dict(data)
        payload = json.loads((output/"references"/f"{name}.json").read_text())
        metric, feas = check_payload(instance, payload)
        assert payload["status"] == "converged" and metric["certificate_score"] <= 1e-8 and feas <= 1e-8
        references[name] = metric["objective"]
    rows = list(csv.DictReader((output/"runs.csv").open()))
    assert len(rows) == len(jobs) == manifest["timed_solves"]
    passes, failures, statuses = Counter(), Counter(), Counter()
    for row, job in zip(rows, jobs):
        for key, value in job.items(): assert str(value) == row[key], (job["job_id"], key)
        payload = json.loads((output/"traces"/f"{job['job_id']}.json").read_text())
        assert payload["job"] == job
        method = job["method"]
        statuses[f"{method}:{row['status']}"] += 1
        if row["status"] == "error":
            assert row["passed"] == "False" and payload["passed"] is False and payload["error"] == row["error"]
            failures[method] += 1
            continue
        metric, feas = check_payload(instances[job["instance"]], payload)
        ref = references[job["instance"]]
        gap = abs(metric["objective"]-ref)/max(1., abs(ref))
        tol = job["tolerance"]
        passed = feas <= tol
        if method != "greedy":
            passed = passed and payload["status"] == "converged" and metric["certificate_score"] <= tol and gap <= 5*tol
            np.testing.assert_allclose(metric["certificate_score"], float(row["certificate_score"]), rtol=1e-10, atol=1e-11)
        assert row["status"] == payload["status"]
        assert int(row["iterations"]) == payload["iterations"]
        assert float(row["seconds"]) == payload["seconds"] >= 0
        assert (row["passed"] == "True") == bool(passed) == payload["passed"]
        for key, value in dict(objective=metric["objective"], reference_objective=ref,
                              reference_gap=gap, business_feasibility=feas, satisfaction=metric["satisfaction"]).items():
            np.testing.assert_allclose(float(row[key]), value, rtol=1e-10, atol=1e-11)
        (passes if passed else failures)[method] += 1
    sensitivity = list(csv.DictReader((output/"sensitivity.csv").open()))
    expected = len(config["seeds"])*2*config["sensitivity_case"][1]*len(config["sensitivity_perturbations"])
    assert len(sensitivity) == expected
    baseline_ids = set()
    for row in sensitivity:
        payloads = []
        for key in ["baseline_id", "trace_id"]:
            payload = json.loads((output/"sensitivity"/f"{row[key]}.json").read_text())
            instance = Instance.from_dict(payload["input"])
            metric, feas = check_payload(instance, payload["solution"])
            assert payload["solution"]["status"] == "converged" and metric["certificate_score"] <= 1e-8 and feas <= 1e-8
            payloads.append(payload)
        baseline_ids.add(row["baseline_id"])
        base, changed = payloads
        resource, j = row["resource"], int(row["rsu"])
        field = resource+"_capacity"
        delta = base["input"][field][j]*float(row["fraction"])
        expected_input = json.loads(json.dumps(base["input"]))
        expected_input[field][j] += delta
        expected_input["name"] = changed["input"]["name"]
        assert expected_input == changed["input"]
        n, m = len(base["input"]["demand"]), len(base["input"]["compute_capacity"])
        price = -base["solution"]["raw_dual"][n+(0 if resource=="compute" else m)+j]
        bcost = base["solution"]["metrics"]["objective"]
        ccost = changed["solution"]["metrics"]["objective"]
        actual, predicted = bcost-ccost, price*delta
        for key, value in dict(delta_capacity=delta, shadow_price=price,
                baseline_objective=bcost, perturbed_objective=ccost,
                actual_cost_reduction=actual, predicted_cost_reduction=predicted,
                absolute_prediction_error=abs(actual-predicted),
                prediction_error_over_baseline=abs(actual-predicted)/max(1., abs(bcost))).items():
            np.testing.assert_allclose(float(row[key]), value, rtol=1e-10, atol=1e-10)
        assert actual >= -1e-6 and predicted >= -1e-6 and actual <= predicted+1e-5
    assert len(sensitivity)+len(baseline_ids) == manifest["sensitivity_solves"]
    report = dict(audit_passed=True, verified_inputs=len(instances),
        verified_timed_solves=len(rows), verified_reference_solves=len(references),
        verified_sensitivity_solves=len(sensitivity)+len(baseline_ids),
        verified_source_hashes=len(manifest["source_sha256"]),
        algorithm_pass_counts=dict(passes), algorithm_nonpass_counts=dict(failures), statuses=dict(statuses),
        note="All physical allocations, raw-unit certificates, reference objectives and labels recomputed. Non-convergence remains recorded as failure.")
    dump(output/"audit.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=BASE/"experiments/formal")
    audit(parser.parse_args().experiment)
