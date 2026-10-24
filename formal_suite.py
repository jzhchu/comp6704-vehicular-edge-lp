"""Pre-specified paired experiments, ablations and raw-unit dual sensitivity."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from algorithms import SOLVERS, make_problem
from benchmark import export_solution, feasibility, write_csv
from generate_instances import generate
from minimal_model import Instance

BASE = Path(__file__).resolve().parent


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_jobs(config):
    inputs, specs = {}, []

    def add(group, n, m, seed, rc=1., rb=1., radius=None, tolerance=None,
            scaling=True, methods=None, iterations=None, repetitions=None, seconds=None):
        instance, data = generate(n, m, rc, rb, config["radius"] if radius is None else radius, seed)
        inputs[instance.name] = data
        for rep in range(config["repetitions"] if repetitions is None else repetitions):
            for method in config["methods"] if methods is None else methods:
                specs.append(dict(group=group, instance=instance.name, vehicles=n, rsus=m,
                    seed=seed, rho_compute=rc, rho_upload=rb,
                    radius=config["radius"] if radius is None else radius,
                    method=method, repetition=rep, scaling=scaling,
                    tolerance=config["tolerance"] if tolerance is None else tolerance,
                    max_iterations=iterations or config["iteration_limits"].get(method, 0),
                    time_limit_seconds=config["time_limit_seconds"] if seconds is None else seconds))

    opt_methods = [m for m in config["methods"] if m != "greedy"]
    for seed in config["seeds"]:
        # Deduplicate identical capacity corner within the application block.
        main = {}
        def main_case(group, n, m, rc=1., rb=1., radius=None):
            key = (n, m, rc, rb, config["radius"] if radius is None else radius)
            main.setdefault(key, []).append(group)
        for n, m in config["scale_cases"]:
            main_case("scale", n, m)
        for ratio in config["capacity_ratios"]:
            n, m = config["capacity_case"]
            main_case("compute", n, m, ratio, config["other_capacity_ratio"])
            main_case("upload", n, m, config["other_capacity_ratio"], ratio)
        for radius in config["radii"]:
            main_case("connectivity", *config["connectivity_case"], radius=radius)
        for (n, m, rc, rb, radius), groups in main.items():
            add("|".join(groups), n, m, seed, rc, rb, radius)
        for n, m in config["precision_cases"]:
            for tol in config["tolerances"]:
                add("precision", n, m, seed, tolerance=tol, methods=opt_methods)
        for n, m in config["ablation_cases"]:
            for scaling in [True, False]:
                add("scaling", n, m, seed, scaling=scaling, methods=opt_methods)
        for n, m in config["budget_cases"]:
            for iterations in config["pdhg_iteration_budgets"]:
                add("budget", n, m, seed, methods=["pdhg"], iterations=iterations,
                    repetitions=1, seconds=config["budget_time_limit_seconds"])
    # Randomized serial execution; no competing timed solver processes.
    order = np.random.default_rng(config["execution_order_seed"]).permutation(len(specs))
    jobs = [dict(job_id=f"j{i:05d}", **specs[int(k)]) for i, k in enumerate(order)]
    return inputs, jobs


def assess(problem, solution, reference, tolerance):
    payload = export_solution(problem, solution)
    x, u = problem.allocation(solution.z)
    disconnected = np.isnan(problem.instance.cost)
    feas = max(feasibility(problem.instance, x, u),
               float(np.max(np.abs(x[disconnected]), initial=0.)))
    metrics = payload["metrics"]
    gap = abs(metrics["objective"] - reference) / max(1., abs(reference))
    passed = feas <= tolerance
    if solution.method != "greedy":
        passed = passed and solution.status == "converged" and metrics["certificate_score"] <= tolerance and gap <= 5*tolerance
    payload.update(reference_objective=reference, reference_gap=gap,
                   business_feasibility=feas, passed=bool(passed))
    return payload


def certified_reference(instance):
    p = make_problem(instance)
    s = SOLVERS["highs"](p, tolerance=1e-8, time_limit=30.)
    payload = assess(p, s, p.metrics(s.z, s.y)["objective"], 1e-8)
    if not payload["passed"]:
        raise RuntimeError(f"Uncertified reference: {instance.name}")
    return payload


PLAN = """车联网边缘计算：正式合成批量实验计划

研究问题
1. 教学版单纯形、内点法和 PDHG 在共同原始单位精度下的通过率与耗时如何随规模变化？
2. 计算容量、上传容量与允许连接范围变化时，最优成本与服务率如何变化？
3. 精度、共同矩阵缩放与 PDHG 计算预算如何影响收敛？
4. 原始单位下的对偶资源价格能否预测小幅增加单台 RSU 容量的成本变化？

模型与数据
单窗口、可拆分连续工作量、给定成本与两类容量。无真实轨迹或通信协议仿真。
所有输入由固定种子生成并保存。容量与半径扫描在相同种子、相同 N/M 下保持其他参数不变。
尺度扫描同时改变 N 与 M，属于系统规模变化，不能据此单独估计 N 或 M 的因果效应。
增加允许连接的边会扩大可行域，LP 最优成本不应上升；成本目标不等价于单独最大化服务率。

预先固定的设计
完整数值配置见 config.json；jobs.json 是求解前生成的随机串行执行清单。
应用主实验：5 个种子，规模 10/30/60/100/200，容量比例 .4/.7/1/1.3/1.6，半径 .1/.2/.35/.5。
五种方法在每个主实验实例重复 3 次，共 1350 次。相同容量角点只运行一次。
精度：N=30/100，目标 1e-3/1e-5/1e-7，四种 LP 方法、5 种子、3 重复，共 360 次。
缩放：同两种规模，四种 LP 方法、缩放开/关、5 种子、3 重复，共 240 次。
该消融关闭列、行、目标共同缩放；PDHG 的固定对角步长仍保留，不能称为完全去除预条件。
预算：PDHG，N=100/200，1万/10万/100万次上限，5 种子，每个配置运行一次，共 30 次。
除预算实验 12 秒外，单次软时限 8 秒；迭代和检查边界决定实际停止，可能略超时。
敏感性：计算/上传分别构造 .7 容量瓶颈，另一资源 1.6；5 种子、4 RSU，单独增加 .1%/1%/5%。
10 次基线与 120 次扰动求解单独记录，不并入 1980 次算法比较。参考解和预热也不并入计时比较。

评价与统计
原始单位的逐行相对残差、对偶不等式残差、非负性、相对原始对偶差距的最大值作为证书分数。
LP 方法还须满足业务可行性、converged 状态，且相对参考目标差距 <= 5*目标精度。
贪心只检查可行性，不能把它的通过率解释为求得最优解。参考解须通过 1e-8 证书验证。
精度扫描改变验证/停止阈值；HiGHS 内部可行性阈值始终 1e-9，不表示自动等预算优化。
计时包括算法准备与迭代监控，不包括输入生成、公共建模、写盘和独立参考求解。
固定数值线程为 1，预热后随机串行执行。教学方法和成熟求解器的日志及预处理开销不同。
先取每实例重复时间中位数，再跨种子取中位数/IQR；失败运行保留并统计，耗时含失败。
贪心相对最优成本差距采用种子为簇的 bootstrap，10000 次重抽样，报告百分位 95% 区间。
同一种子的各配置共同重抽，避免把重复运行和配对配置误当作独立样本。只有 5 个簇，区间仅作描述。
对偶价格是局部边际值；有限扰动可切换最优基。报告实际与线性预测差值，不要求 5% 扰动仍精确。

可追溯性与限制
输出源码快照、输入哈希、环境、逐次变量/对偶/收敛历史、独立复核及图表。
不删除超时、迭代上限或数值失败；不使用参考最优解初始化教学算法。
不测内存、不运行真实车辆数据、不据此声称工业可扩展性。随机工作负载下结论仅适用于该生成分布。
"""


def run_sensitivity(config, output):
    rows = []
    n, m = config["sensitivity_case"]
    for seed in config["seeds"]:
        for resource in ["compute", "upload"]:
            rc, rb = config["sensitivity_bottleneck_ratio"], config["other_capacity_ratio"]
            if resource == "upload": rc, rb = rb, rc
            instance, data = generate(n, m, rc, rb, config["radius"], seed)
            baseline = certified_reference(instance)
            base_id = f"{resource}_s{seed}_base"
            dump(output / "sensitivity" / f"{base_id}.json", dict(input=data, solution=baseline))
            objective = baseline["metrics"]["objective"]
            offset = n + (0 if resource == "compute" else m)
            for j in range(m):
                price = -baseline["raw_dual"][offset+j]
                field = resource + "_capacity"
                for fraction in config["sensitivity_perturbations"]:
                    changed = json.loads(json.dumps(data))
                    delta = data[field][j]*fraction
                    changed[field][j] += delta
                    changed["name"] += f"__{resource}{j}_plus{fraction:g}"
                    solved = certified_reference(Instance.from_dict(changed))
                    trace_id = f"{resource}_s{seed}_j{j}_f{fraction:g}"
                    dump(output / "sensitivity" / f"{trace_id}.json", dict(input=changed, solution=solved))
                    actual = objective-solved["metrics"]["objective"]
                    prediction = price*delta
                    rows.append(dict(trace_id=trace_id, baseline_id=base_id, seed=seed,
                        resource=resource, rsu=j, fraction=fraction, delta_capacity=delta,
                        shadow_price=price, baseline_objective=objective,
                        perturbed_objective=solved["metrics"]["objective"],
                        actual_cost_reduction=actual, predicted_cost_reduction=prediction,
                        absolute_prediction_error=abs(actual-prediction),
                        prediction_error_over_baseline=abs(actual-prediction)/max(1., abs(objective))))
    write_csv(output / "sensitivity.csv", rows)
    return len(rows) + 2*len(config["seeds"])


def run(config, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    inputs, jobs = build_jobs(config)
    dump(output / "config.json", config)
    dump(output / "jobs.json", jobs)
    (output / "plan.txt").write_text(PLAN, encoding="utf-8")
    # Freeze the runnable code before any experiment, including dependencies.
    snapshot = output / "source_snapshot"
    sources = list(BASE.glob("*.py")) + list(BASE.glob("*.sh")) + [BASE/"requirements.txt"]
    sources += list((BASE / "tests").glob("*.py"))
    source_hashes = {}
    for path in sources:
        relative = path.relative_to(BASE)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        source_hashes[str(relative)] = sha(target)
    dump(snapshot / "configs/formal_suite.json", config)
    (snapshot / "data").mkdir(exist_ok=True)
    (snapshot / "data/two_vehicles.json").write_bytes((BASE / "data/two_vehicles.json").read_bytes())
    instances, hashes, references = {}, {}, {}
    for name, data in inputs.items():
        path = output / "instances" / f"{name}.json"
        dump(path, data); hashes[name] = sha(path)
        instances[name] = Instance.from_dict(data)
        references[name] = certified_reference(instances[name])
        dump(output / "references" / f"{name}.json", references[name])
    warm = make_problem(Instance.from_dict(json.loads((BASE / "data/two_vehicles.json").read_text())))
    for method in config["methods"]:
        SOLVERS[method](warm)
    problems = {(name, scaling): make_problem(instance, scaling=scaling)
                for name, instance in instances.items() for scaling in [True, False]}
    rows = []
    for i, job in enumerate(jobs, 1):
        problem = problems[job["instance"], job["scaling"]]
        ref = references[job["instance"]]["metrics"]["objective"]
        row = dict(job, status="error", passed=False, seconds=None, iterations=None,
                   objective=None, reference_objective=ref, reference_gap=None,
                   certificate_score=None, business_feasibility=None, satisfaction=None, error="")
        try:
            kwargs = dict(tolerance=job["tolerance"], time_limit=job["time_limit_seconds"])
            if job["max_iterations"]: kwargs["max_iterations"] = job["max_iterations"]
            solution = SOLVERS[job["method"]](problem, **kwargs)
            payload = assess(problem, solution, ref, job["tolerance"])
            metric = payload["metrics"]
            row.update(status=solution.status, passed=payload["passed"], seconds=solution.seconds,
                iterations=solution.iterations, objective=metric["objective"],
                reference_gap=payload["reference_gap"], business_feasibility=payload["business_feasibility"],
                certificate_score=metric["certificate_score"] if job["method"] != "greedy" else None,
                satisfaction=metric["satisfaction"])
            payload["job"] = job
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            payload = dict(job=job, error=row["error"], passed=False)
        dump(output / "traces" / f"{job['job_id']}.json", payload)
        rows.append(row)
        if i % 25 == 0 or i == len(jobs):
            write_csv(output / "runs.csv", rows)
            print(f"[{i}/{len(jobs)}] recorded; failures so far: " + str(dict(Counter(r["method"] for r in rows if not r["passed"]))), flush=True)
    sensitivity_solves = run_sensitivity(config, output)
    manifest = dict(config=config, started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        timed_solves=len(rows), reference_solves=len(references), sensitivity_solves=sensitivity_solves,
        instance_count=len(instances), source_snapshot="source_snapshot", source_sha256=source_hashes,
        instance_sha256=hashes, jobs_sha256=sha(output/"jobs.json"), config_sha256=sha(output/"config.json"),
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
            platform=platform.platform(), machine=platform.machine(),
            thread_limits={k:os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]}))
    manifest["result_sha256"] = {str(path.relative_to(output)): sha(path)
        for path in output.rglob("*") if path.is_file() and "source_snapshot" not in path.parts
        and "instances" not in path.parts}
    dump(output / "manifest.json", manifest)
    print(f"Complete: {len(rows)} timed solves, {len(references)} references, {sensitivity_solves} sensitivity solves.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE/"configs/formal_suite.json")
    parser.add_argument("--output", type=Path, default=BASE/"experiments/formal")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(seeds=[11], repetitions=1, scale_cases=[[10,3]],
            capacity_case=[10,3], capacity_ratios=[.7,1.3], connectivity_case=[10,3], radii=[.15,.4],
            precision_cases=[[10,3]], ablation_cases=[[10,3]], budget_cases=[[10,3]],
            pdhg_iteration_budgets=[50,500], sensitivity_case=[10,3])
    run(config, args.output)


if __name__ == "__main__":
    main()
