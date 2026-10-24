"""Run paired, seeded comparisons; preserve failed and timed-out solves."""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from algorithms import SOLVERS, make_problem
from generate_instances import generate
from minimal_model import Instance, validate_solution

BASE = Path(__file__).resolve().parent


def cases(config):
    records = {}

    def add(n, m, rc, rb, seed, group):
        instance, data = generate(n, m, rc, rb, config["radius"], seed)
        if instance.name not in records:
            records[instance.name] = {"instance": instance, "data": data, "groups": [],
                "n": n, "m": m, "rho_compute": rc, "rho_upload": rb, "seed": seed}
        records[instance.name]["groups"].append(group)

    for seed in config["seeds"]:
        for n, m in config["scale_cases"]:
            add(n, m, 1.0, 1.0, seed, "scale")
        for ratio in config["capacity_ratios"]:
            add(config["capacity_vehicle_count"], config["capacity_rsu_count"], ratio,
                config["other_capacity_ratio"], seed, "compute")
            add(config["capacity_vehicle_count"], config["capacity_rsu_count"],
                config["other_capacity_ratio"], ratio, seed, "upload")
    if config["include_analytical_example"]:
        data = json.loads((BASE / "data/two_vehicles.json").read_text())
        instance = Instance.from_dict(data)
        records[instance.name] = {"instance": instance, "data": data, "groups": ["analytical"],
                                  "n": 2, "m": 2, "rho_compute": 1, "rho_upload": 1, "seed": -1}
    return list(records.values())


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def export_solution(problem, solution):
    raw, dual = problem.original(solution.z, solution.y)
    x, u = problem.allocation(solution.z)
    return {"method": solution.method, "status": solution.status,
            "iterations": solution.iterations, "seconds": solution.seconds,
            "metrics": problem.metrics(solution.z, solution.y), "allocation": x.tolist(),
            "unserved": u.tolist(), "raw_primal": raw.tolist(), "raw_dual": dual.tolist(),
            "business_audit": validate_solution(problem.instance, x, u),
            "details": solution.details, "history": solution.history}


def feasibility(instance, x, u):
    compute = (instance.compute_per_workload[:, None] * x).sum(axis=0)
    upload = (instance.upload_per_workload[:, None] * x).sum(axis=0)
    violations = [np.max(abs(x.sum(axis=1) + u - instance.demand) / (1 + instance.demand)),
                  np.max(np.maximum(compute - instance.compute_capacity, 0) / (1 + instance.compute_capacity)),
                  np.max(np.maximum(upload - instance.upload_capacity, 0) / (1 + instance.upload_capacity)),
                  max(0, -float(x.min()), -float(u.min()))]
    return float(max(violations))


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["instance"], row["method"]), []).append(row)
    output = []
    for (name, method), trials in groups.items():
        times = [r["seconds"] for r in trials if r["seconds"] is not None]
        valid = [r for r in trials if r["passed"]]
        gaps = [r["reference_gap"] for r in valid]
        output.append({"instance": name, "method": method, "groups": trials[0]["groups"],
            "vehicles": trials[0]["vehicles"], "rsus": trials[0]["rsus"], "seed": trials[0]["seed"],
            "rho_compute": trials[0]["rho_compute"], "rho_upload": trials[0]["rho_upload"],
            "repetitions": len(trials), "passed": len(valid),
            "seconds_median_all_runs": float(np.median(times)) if times else None,
            "seconds_q25_all_runs": float(np.quantile(times, 0.25)) if times else None,
            "seconds_q75_all_runs": float(np.quantile(times, 0.75)) if times else None,
            "reference_gap_median_passed": float(np.median(gaps)) if gaps else None,
            "satisfaction_median_passed": float(np.median([r["satisfaction"] for r in valid])) if valid else None})
    return output


def summary(config, rows, grouped, case_count):
    lines = ["车联网边缘计算算法比较实验", "", f"实例数：{case_count}（包含解析例子）",
             f"每种方法每实例重复：{config['repetitions']} 次", f"计时求解次数：{len(rows)}",
             f"共同目标精度：{config['tolerance']:g}；单次时限：{config['time_limit_seconds']:g} 秒", "",
             "方法结果（贪心只检查可行性；其他方法同时检查 KKT、可行性与参考目标差距）"]
    for method in config["methods"]:
        trial = [r for r in rows if r["method"] == method]
        passed = [r for r in trial if r["passed"]]
        complete_instances = [r for r in grouped if r["method"] == method and r["passed"] == config["repetitions"]]
        times = [r["seconds"] for r in trial if r["seconds"] is not None]
        gap = max([r["reference_gap"] for r in passed], default=None)
        lines.append(f"{method}: 通过 {len(passed)}/{len(trial)} 次；全部重复均通过的实例 {len(complete_instances)}/{case_count}；"
                     f"所有有计时记录运行的中位耗时 {np.median(times):.6f} 秒；"
                     + (f"通过运行的最大参考目标差距 {gap:.3g}" if gap is not None else "无有效差距"))
        failures = sorted({r["status"] for r in trial if not r["passed"]})
        if failures: lines.append("  未通过状态：" + ", ".join(failures))
    analytical = [r for r in rows if r["groups"] == "analytical"]
    lines.extend(["", "两辆车解析例子的实际结果"])
    for method in config["methods"]:
        values = [r["objective"] for r in analytical if r["method"] == method and r["objective"] is not None]
        if values: lines.append(f"{method}: 成本中位数 {np.median(values):.8f}")
    greedy = [r for r in grouped if r["method"] == "greedy" and r["passed"] and r["groups"] != "analytical"]
    if greedy:
        gaps = [r["reference_gap_median_passed"] for r in greedy]
        lines.extend(["", f"合成随机实例中，贪心的相对参考成本差距中位数为 {np.median(gaps):.2%}，范围 {min(gaps):.2%} 到 {max(gaps):.2%}。"])
    lines.extend(["", "实验解释与范围",
        "1. 三种自编 LP 算法共用相同标准形式、缩放、原始单位检查和停止精度。贪心不声称最优。",
        "2. 时间包含算法内部准备和日志检查，不包含实例生成与公共模型构建。线程数固定为 1；每实例先对重复取中位数，再比较不同种子。",
        "3. 超时、迭代上限、证书失败和异常均保留在 runs.csv 中。耗时汇总不删除失败运行；成功比例单独报告。",
        "4. HiGHS 使用成熟实现与预处理，且只在结束时检查；教学实现记录中间日志。时间差包含实现和监控开销，不能单独归因于算法理论。",
        "5. IPM 采用 Mehrotra predictor-corrector；PDHG 使用固定对角预条件步长，不是完整 PDLP；这些细节与早期方案的基础版本有所不同。",
        "6. 这是固定已知参数、可拆分工作量、单窗口合成实验。未验证真实车辆轨迹、端到端时延、跨窗口调度或大规模工业性能。",
        "7. 当前批次以规模与计算/通信容量为变量；尚未完成全部精度扫参、参数消融、bootstrap 区间或内存测量。",
        "", "文件说明",
        "runs.csv：每次计时求解及失败状态；aggregate.csv：每实例每方法的重复汇总。",
        "instances/：全部输入；traces/：每次求解的变量、对偶、残差和迭代历史。",
        "manifest.json：实际环境、配置、输入哈希与数据生成种子；figures/：结果图。"])
    return "\n".join(lines) + "\n"


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "instances").mkdir(exist_ok=True)
    (output / "traces").mkdir(exist_ok=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    source_hashes = {}
    for name in ["algorithms.py", "generate_instances.py", "minimal_model.py", "benchmark.py"]:
        source = (BASE / name).read_bytes()
        (snapshot / name).write_bytes(source)
        source_hashes[name] = hashlib.sha256(source).hexdigest()
    (snapshot / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (snapshot / "configs").mkdir(exist_ok=True)
    (snapshot / "configs/benchmark.json").write_text(json.dumps(config, indent=2) + "\n")
    (snapshot / "data").mkdir(exist_ok=True)
    (snapshot / "data/two_vehicles.json").write_bytes((BASE / "data/two_vehicles.json").read_bytes())
    if (BASE / "requirements.txt").exists():
        (snapshot / "requirements.txt").write_bytes((BASE / "requirements.txt").read_bytes())
    records = cases(config)
    order_rng = np.random.default_rng(config["execution_order_seed"])
    # Warm up imports, linear algebra and solver initialization outside timing.
    warm = make_problem(Instance.from_dict(json.loads((BASE / "data/two_vehicles.json").read_text())))
    for method in config["methods"]:
        SOLVERS[method](warm, tolerance=config["tolerance"], time_limit=config["time_limit_seconds"])
    rows, hashes = [], {}
    for index, record in enumerate(records, 1):
        instance = record["instance"]
        source = json.dumps(record["data"], indent=2, allow_nan=False) + "\n"
        (output / "instances" / (instance.name + ".json")).write_text(source)
        hashes[instance.name] = hashlib.sha256(source.encode()).hexdigest()
        problem = make_problem(instance)
        reference = SOLVERS["highs"](problem, tolerance=config["tolerance"], time_limit=config["time_limit_seconds"])
        ref_cost = problem.metrics(reference.z, reference.y)["objective"]
        for rep in range(config["repetitions"]):
            for method in order_rng.permutation(config["methods"]).tolist():
                row = {"instance": instance.name, "groups": "|".join(record["groups"]),
                    "vehicles": record["n"], "rsus": record["m"], "seed": record["seed"],
                    "rho_compute": record["rho_compute"], "rho_upload": record["rho_upload"],
                    "method": method, "repetition": rep, "status": "error", "passed": False,
                    "seconds": None, "iterations": None, "objective": None, "reference_objective": ref_cost,
                    "reference_gap": None, "certificate_score": None, "business_feasibility": None,
                    "absolute_business_violation": None, "satisfaction": None, "error": ""}
                try:
                    solution = SOLVERS[method](problem, tolerance=config["tolerance"], time_limit=config["time_limit_seconds"])
                    payload = export_solution(problem, solution)
                    metrics = payload["metrics"]
                    x, u = problem.allocation(solution.z)
                    feas = feasibility(instance, x, u)
                    gap = abs(metrics["objective"] - ref_cost) / max(1, abs(ref_cost))
                    passed = feas <= config["tolerance"]
                    if method != "greedy":
                        passed = passed and solution.status == "converged" and metrics["certificate_score"] <= config["tolerance"] and gap <= 5 * config["tolerance"]
                    row.update(status=solution.status, passed=bool(passed), seconds=solution.seconds,
                        iterations=solution.iterations, objective=metrics["objective"], reference_gap=gap,
                        certificate_score=metrics["certificate_score"] if method != "greedy" else None,
                        business_feasibility=feas, absolute_business_violation=metrics["absolute_business_violation"],
                        satisfaction=metrics["satisfaction"])
                    payload.update(reference_objective=ref_cost, reference_gap=gap, passed=bool(passed),
                                   business_feasibility=feas, repetition=rep, instance=instance.name)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    payload = dict(row)
                (output / "traces" / f"{instance.name}__{method}__r{rep}.json").write_text(
                    json.dumps(payload, indent=2, allow_nan=False) + "\n")
                rows.append(row)
        print(f"[{index}/{len(records)}] {instance.name}: " + ", ".join(
            f"{method} {sum(r['passed'] for r in rows if r['instance'] == instance.name and r['method'] == method)}/{config['repetitions']}"
            for method in config["methods"]), flush=True)
        write_csv(output / "runs.csv", rows)
    grouped = aggregate(rows)
    write_csv(output / "aggregate.csv", grouped)
    manifest = {"config": config, "instance_count": len(records), "timed_solves": len(rows),
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform(), "machine": platform.machine(),
                        "thread_limits": {key: os.environ.get(key) for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]}},
        "instance_sha256": hashes,
        "source_sha256": source_hashes, "source_snapshot": "source_snapshot",
        "effective_iteration_limits": {name: inspect.signature(SOLVERS[name]).parameters["max_iterations"].default
                                       for name in ["simplex", "ipm", "pdhg"]}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "summary.txt").write_text(summary(config, rows, grouped, len(records)), encoding="utf-8")
    print((output / "summary.txt").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE / "configs/benchmark.json")
    parser.add_argument("--output", type=Path, default=BASE / "experiments/batch")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.pilot:
        config.update(seeds=[11], repetitions=1, scale_cases=[[10, 3], [30, 4]],
                      capacity_ratios=[0.6, 1.4])
    run(config, args.output)


if __name__ == "__main__":
    main()
