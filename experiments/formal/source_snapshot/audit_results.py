"""Check saved inputs, source hashes and every reported solve without re-solving."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from algorithms import make_problem
from benchmark import feasibility
from minimal_model import Instance, validate_solution

BASE = Path(__file__).resolve().parent


def audit(experiment):
    manifest = json.loads((experiment / "manifest.json").read_text())
    config = manifest["config"]
    source_dir = experiment / manifest.get("source_snapshot", ".")
    if "source_snapshot" not in manifest:
        source_dir = BASE
    for name, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        if actual != expected: raise ValueError(f"Source hash mismatch: {name}")
    instances = {}
    for name, expected in manifest["instance_sha256"].items():
        path = experiment / "instances" / (name + ".json")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input hash mismatch: {name}")
        instances[name] = Instance.from_dict(json.loads(path.read_text()))
    rows = list(csv.DictReader((experiment / "runs.csv").open()))
    if len(rows) != manifest["timed_solves"]:
        raise ValueError("Timed-solve count mismatch")
    by_method, nonpasses = Counter(), Counter()
    keys = set()
    for row in rows:
        key = (row["instance"], row["method"], row["repetition"])
        if key in keys: raise ValueError(f"Duplicate result: {key}")
        keys.add(key)
        trace = json.loads((experiment / "traces" / f"{key[0]}__{key[1]}__r{key[2]}.json").read_text())
        if row["status"] == "error":
            if row["passed"] != "False" or not trace.get("error"):
                raise ValueError(f"Unrecorded solver failure: {key}")
            nonpasses[row["method"]] += 1
            continue
        instance = instances[row["instance"]]
        p = make_problem(instance)
        raw, dual = np.array(trace["raw_primal"]), np.array(trace["raw_dual"])
        z, y = raw / p.column_scale, dual / (p.cost_scale * p.row_scale)
        x, u = p.allocation(z)
        np.testing.assert_allclose(x, trace["allocation"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(u, trace["unserved"], rtol=1e-12, atol=1e-12)
        check = validate_solution(instance, x, u)
        metrics = p.metrics(z, y)
        np.testing.assert_allclose(metrics["objective"], float(row["objective"]), rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(check["objective"], float(row["objective"]), rtol=1e-12, atol=1e-10)
        feas = feasibility(instance, x, u)
        gap = abs(metrics["objective"] - float(row["reference_objective"])) / max(1, abs(float(row["reference_objective"])))
        passed = feas <= config["tolerance"]
        if row["method"] != "greedy":
            passed = passed and row["status"] == "converged" and metrics["certificate_score"] <= config["tolerance"] and gap <= 5 * config["tolerance"]
        if bool(passed) != (row["passed"] == "True") or bool(passed) != trace["passed"]:
            raise ValueError(f"Incorrect pass/fail label: {key}")
        if passed: by_method[row["method"]] += 1
        else: nonpasses[row["method"]] += 1
    expected = len(instances) * config["repetitions"] * len(config["methods"])
    if len(keys) != expected: raise ValueError("Missing paired solves")
    report = {"audit_passed": True, "verified_inputs": len(instances),
              "verified_source_hashes": len(manifest["source_sha256"]), "verified_timed_solves": len(rows),
              "algorithm_pass_counts": dict(by_method), "algorithm_nonpass_counts": dict(nonpasses),
              "note": "Audit validates recorded data and labels; non-converged solves remain non-converged."}
    (experiment / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=BASE / "experiments/batch")
    audit(parser.parse_args().experiment)
