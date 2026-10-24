"""Create paper figures and descriptive statistics from audited saved results."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE.parents[1]/"work/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import write_csv
from formal_suite import dump
from plot_results import NAMES, COLORS, save

MAIN_GROUPS = {"scale", "compute", "upload", "compute|upload", "connectivity"}
METHODS = ["highs", "simplex", "ipm", "pdhg", "greedy"]


def load_rows(output):
    rows = list(csv.DictReader((output/"runs.csv").open()))
    for row in rows:
        for key in ["vehicles", "rsus", "seed", "repetition", "max_iterations", "iterations"]:
            row[key] = int(row[key]) if row[key] else None
        for key in ["rho_compute", "rho_upload", "radius", "tolerance", "time_limit_seconds", "seconds", "objective",
                    "reference_objective", "reference_gap", "certificate_score", "business_feasibility", "satisfaction"]:
            row[key] = float(row[key]) if row[key] else None
        row["passed"] = row["passed"] == "True"
        row["scaling"] = row["scaling"] == "True"
    return rows


def per_instance(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ["group", "instance", "method", "tolerance", "scaling", "max_iterations", "time_limit_seconds"])
        groups[key].append(row)
    output = []
    for trials in groups.values():
        first = trials[0]
        row = {k:first[k] for k in ["group", "instance", "vehicles", "rsus", "seed", "rho_compute", "rho_upload",
                                   "radius", "method", "tolerance", "scaling", "max_iterations", "time_limit_seconds"]}
        valid = [r for r in trials if r["passed"]]
        row.update(trials=len(trials), passes=len(valid))
        for metric in ["seconds", "certificate_score"]:
            values = [r[metric] for r in trials if r[metric] is not None]
            row[metric] = float(np.median(values)) if values else None
        for metric in ["objective", "reference_gap", "satisfaction"]:
            values = [r[metric] for r in valid]
            row[metric] = float(np.median(values)) if values else None
        output.append(row)
    return output


def cell_summary(instances):
    groups = defaultdict(list)
    fields = ["group", "vehicles", "rsus", "rho_compute", "rho_upload", "radius", "method", "tolerance", "scaling", "max_iterations", "time_limit_seconds"]
    for row in instances: groups[tuple(row[k] for k in fields)].append(row)
    output = []
    for key, observations in groups.items():
        row = dict(zip(fields, key))
        row.update(seeds=len(observations), trials=sum(r["trials"] for r in observations),
                   passes=sum(r["passes"] for r in observations))
        for metric in ["seconds", "certificate_score", "objective", "reference_gap", "satisfaction"]:
            values = [r[metric] for r in observations if r[metric] is not None]
            for q, label in [(.5,"median"), (.25,"q25"), (.75,"q75")]:
                row[f"{metric}_{label}"] = float(np.quantile(values, q)) if values else None
        output.append(row)
    return output


def center_iqr(observations, field):
    values = [r[field] for r in observations if r[field] is not None]
    center = float(np.median(values))
    return center, center-float(np.quantile(values,.25)), float(np.quantile(values,.75))-center


def error_curve(ax, xs, select, field, label, color, **kwargs):
    stats = [center_iqr(select(x), field) for x in xs]
    ax.errorbar(xs, [r[0] for r in stats], yerr=[[r[1] for r in stats],[r[2] for r in stats]],
        label=label, color=color, marker="o", capsize=3, markersize=4, linewidth=1.5, **kwargs)


def figures(output, config, rows, instances, sensitivity):
    dest = output/"figures"; dest.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.spines.top":False, "axes.spines.right":False, "legend.frameon":False})
    nseeds = len(config["seeds"])
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    sizes = [n for n,m in config["scale_cases"]]
    for method in METHODS:
        select = lambda n: [r for r in instances if r["group"]=="scale" and r["vehicles"]==n and r["method"]==method]
        error_curve(axes[0], sizes, select, "seconds", NAMES[method], COLORS[method])
        rates = [sum(r["passes"] for r in select(n))/sum(r["trials"] for r in select(n)) for n in sizes]
        axes[1].plot(sizes, np.array(rates)*100, marker="o", label=NAMES[method], color=COLORS[method])
    axes[0].set_yscale("log"); axes[0].set_ylabel("Time (seconds, log scale)")
    axes[0].set_title("A  Runtime including unsuccessful solves"); axes[0].legend(fontsize=8)
    axes[1].set_title("B  Validation pass rate"); axes[1].set_ylabel("Passed trials (%)"); axes[1].set_ylim(-5,105)
    for ax in axes: ax.set_xlabel("Vehicles (RSUs grow with vehicles)"); ax.set_xticks(sizes); ax.grid(axis="y",alpha=.2)
    fig.suptitle(f"System size: tolerance {config['tolerance']:g}, {nseeds} seeds, {config['repetitions']} repetitions")
    fig.supxlabel("Time: median of within-instance medians; bars: IQR across seeds. Greedy is tested for feasibility only.",fontsize=9)
    save(fig,dest,"01_scale")

    fig, axes = plt.subplots(2,3,figsize=(14,7),layout="constrained")
    for col, (group, field, xlabel) in enumerate([("compute","rho_compute","Compute capacity / aggregate demand"),
            ("upload","rho_upload","Upload capacity / aggregate demand"), ("connectivity","radius","Connection radius (unit road)")]):
        selected = [r for r in instances if group in r["group"].split("|")]
        xs = sorted({r[field] for r in selected})
        for method in ["highs","greedy"]:
            choose = lambda x:[r for r in selected if r[field]==x and r["method"]==method]
            error_curve(axes[0,col],xs,choose,"objective",NAMES[method],COLORS[method])
            error_curve(axes[1,col],xs,choose,"satisfaction",NAMES[method],COLORS[method])
        axes[0,col].set_title(group.capitalize()); axes[0,col].set_ylabel("Cost + unserved penalty")
        axes[1,col].set_ylabel("Served workload / demand"); axes[1,col].set_ylim(0,1.04)
        for ax in axes[:,col]: ax.set_xlabel(xlabel); ax.set_xticks(xs); ax.grid(axis="y",alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Paired application sweeps: capacities at N=30/M=4; connectivity at N=60/M=6")
    fig.supxlabel(f"Median and IQR across {nseeds} seeds. Other capacity ratio = {config['other_capacity_ratio']:g}; connectivity uses both ratios = 1.",fontsize=9)
    save(fig,dest,"02_resources_connectivity")

    fig, axes = plt.subplots(len(config["precision_cases"]),2,figsize=(12,7),layout="constrained",squeeze=False)
    xs = sorted(config["tolerances"], reverse=True)
    for index,(n,m) in enumerate(config["precision_cases"]):
        for method in METHODS[:-1]:
            select = lambda t:[r for r in instances if r["group"]=="precision" and r["vehicles"]==n and r["method"]==method and r["tolerance"]==t]
            error_curve(axes[index,0],xs,select,"seconds",NAMES[method],COLORS[method])
            rates = [100*sum(r["passes"] for r in select(t))/sum(r["trials"] for r in select(t)) for t in xs]
            axes[index,1].plot(xs,rates,marker="o",color=COLORS[method],label=NAMES[method])
        axes[index,0].set_yscale("log"); axes[index,0].set_ylabel(f"N={n}, M={m}\nTime (seconds)")
        axes[index,1].set_ylim(-5,105); axes[index,1].set_ylabel("Passed trials (%)")
        for ax in axes[index]: ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel("Target certificate tolerance (stricter to the right)"); ax.grid(axis="y",alpha=.2)
    axes[0,0].set_title("A  Runtime, including failures"); axes[0,1].set_title("B  Validation pass rate"); axes[0,0].legend(fontsize=8)
    fig.suptitle("Accuracy sweep with fixed iteration and time limits")
    fig.supxlabel("HiGHS internal feasibility tolerances remain 1e-9. Time bars: IQR across seeds after within-instance medians.",fontsize=9)
    save(fig,dest,"03_precision")

    fig, axes = plt.subplots(len(config["ablation_cases"]),2,figsize=(12,7),layout="constrained",squeeze=False)
    for index,(n,m) in enumerate(config["ablation_cases"]):
        for offset,scaling in [(-.18,True),(.18,False)]:
            times,rates=[],[]
            for method in METHODS[:-1]:
                select=[r for r in instances if r["group"]=="scaling" and r["vehicles"]==n and r["method"]==method and r["scaling"]==scaling]
                times.append(center_iqr(select,"seconds")[0]); rates.append(100*sum(r["passes"] for r in select)/sum(r["trials"] for r in select))
            for ax, values in zip(axes[index],[times,rates]):
                ax.bar(np.arange(4)+offset,values,width=.35,label="Scaling on" if scaling else "Scaling off",color="#2676B8" if scaling else "#D98238")
        axes[index,0].set_yscale("log"); axes[index,0].set_ylabel(f"N={n}, M={m}\nMedian time (seconds)")
        axes[index,1].set_ylabel("Passed trials (%)"); axes[index,1].set_ylim(0,110)
        for ax in axes[index]: ax.set_xticks(range(4),["HiGHS","Simplex","IPM","PDHG"]); ax.grid(axis="y",alpha=.2)
    axes[0,0].legend(); axes[0,0].set_title("A  Runtime, including failures"); axes[0,1].set_title("B  Validation pass rate")
    fig.suptitle("Common row / column / objective scaling ablation")
    fig.supxlabel("PDHG diagonal step-size preconditioning is retained in both conditions. Common validation uses physical units.",fontsize=9)
    save(fig,dest,"04_scaling_ablation")

    fig, axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    xs=config["pdhg_iteration_budgets"]
    for (n,m),color in zip(config["budget_cases"],["#2676B8","#C55A11"]):
        select=lambda k:[r for r in instances if r["group"]=="budget" and r["vehicles"]==n and r["max_iterations"]==k]
        rates=[100*sum(r["passes"] for r in select(k))/sum(r["trials"] for r in select(k)) for k in xs]
        axes[0].plot(xs,rates,marker="o",color=color,label=f"N={n}, M={m}")
        error_curve(axes[1],xs,select,"certificate_score",f"N={n}, M={m}",color)
    axes[0].set_ylim(-5,105); axes[0].set_ylabel("Passed trials (%)"); axes[0].set_title("A  Validation pass rate"); axes[0].legend()
    axes[1].set_yscale("log"); axes[1].set_ylabel("Best returned certificate score"); axes[1].set_title("B  Certificate at stopping")
    axes[1].axhline(config["tolerance"],linestyle="--",color="#555555",label="Target tolerance"); axes[1].legend(fontsize=8)
    for ax in axes: ax.set_xscale("log"); ax.set_xlabel("Maximum iterations (solver may stop earlier)"); ax.grid(axis="y",alpha=.2)
    fig.suptitle(f"PDHG budget sweep: {nseeds} seeds, one run per configuration, {config['budget_time_limit_seconds']:g}s soft limit")
    fig.supxlabel("Failure states are retained. Bars show IQR across seeds; larger budgets do not guarantee convergence.",fontsize=9)
    save(fig,dest,"05_pdhg_budget")

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    fractions=config["sensitivity_perturbations"]
    for fraction,color in zip(fractions,["#2676B8","#008878","#C55A11"]):
        subset=[r for r in sensitivity if r["fraction"]==fraction]
        axes[0].scatter([r["predicted_cost_reduction"] for r in subset],[r["actual_cost_reduction"] for r in subset],label=f"+{fraction:.1%} capacity",color=color,s=24,alpha=.65)
    limit=max(r["predicted_cost_reduction"] for r in sensitivity)*1.04
    axes[0].plot([0,limit],[0,limit],linestyle="--",color="#666666",label="Exact local prediction")
    axes[0].set_xlabel("Predicted cost reduction: shadow price x added capacity"); axes[0].set_ylabel("Actual cost reduction"); axes[0].legend(fontsize=8); axes[0].set_title("A  Marginal prediction versus re-optimization")
    axes[1].boxplot([[max(1e-14,r["prediction_error_over_baseline"]) for r in sensitivity if r["fraction"]==f] for f in fractions],tick_labels=[f"+{f:.1%}" for f in fractions],showfliers=True)
    axes[1].set_yscale("log"); axes[1].set_ylabel("Absolute prediction error / baseline cost"); axes[1].set_xlabel("Increase in one RSU resource capacity"); axes[1].set_title("B  Prediction error (floor at 1e-14 for display)")
    fig.suptitle("Dual sensitivity: physical-unit prices, compute and upload bottlenecks")
    fig.supxlabel("Prices describe a local slope. Finite capacity increases can change the active constraints and reduce the realized benefit.",fontsize=9)
    save(fig,dest,"06_dual_sensitivity")


def analyze(output):
    audit=json.loads((output/"audit.json").read_text())
    if not audit["audit_passed"]: raise ValueError("Audit must pass before analysis")
    config=json.loads((output/"config.json").read_text())
    rows=load_rows(output); instances=per_instance(rows); cells=cell_summary(instances)
    write_csv(output/"aggregate.csv",instances); write_csv(output/"cell_summary.csv",cells)
    sensitivity=list(csv.DictReader((output/"sensitivity.csv").open()))
    for row in sensitivity:
        for key in row:
            if key not in ["trace_id","baseline_id","resource"]: row[key]=float(row[key])
    greedy=[r for r in instances if r["group"] in MAIN_GROUPS and r["method"]=="greedy" and r["passes"]==r["trials"]]
    seed_means=[float(np.mean([r["reference_gap"] for r in greedy if r["seed"]==seed])) for seed in config["seeds"]]
    rng=np.random.default_rng(config["bootstrap_seed"])
    bootstrap=np.mean(rng.choice(seed_means,size=(config["bootstrap_samples"],len(seed_means)),replace=True),axis=1)
    ci=np.quantile(bootstrap,[.025,.975]).tolist()
    statistics=dict(greedy_mean_relative_cost_gap=float(np.mean(seed_means)),
        greedy_median_relative_cost_gap=float(np.median([r["reference_gap"] for r in greedy])),
        greedy_seed_cluster_bootstrap_95_percentile_interval=ci,
        independent_seed_clusters=len(seed_means), bootstrap_samples=config["bootstrap_samples"],
        bootstrap_seed=config["bootstrap_seed"],
        note="Seeds are resampled as clusters; repeated trials and paired configurations are not treated as independent observations. Five clusters provide descriptive uncertainty only.")
    dump(output/"statistics.json",statistics)
    lines=["车联网边缘计算：正式批量实验结果", "",
        f"输入实例 {len({r['instance'] for r in rows})} 个；算法比较 {len(rows)} 次；敏感性求解 {audit['verified_sensitivity_solves']} 次。",
        f"固定种子 {config['seeds']}；一般配置重复 {config['repetitions']} 次，PDHG 预算实验每配置一次。",
        "五种方法：HiGHS、教学版单纯形、教学版内点法、固定对角步长 PDHG、可行贪心。", "",
        "一、应用主实验（共同精度 1e-5；贪心仅检查可行性）"]
    main=[r for r in rows if r["group"] in MAIN_GROUPS]
    for method in METHODS:
        subset=[r for r in main if r["method"]==method]
        good=sum(r["passed"] for r in subset)
        lines.append(f"{method}: {good}/{len(subset)} 通过；失败状态 {dict(Counter(r['status'] for r in subset if not r['passed']))}。")
    lines += [f"贪心相对最优成本差距：跨配置/种子的平均值 {statistics['greedy_mean_relative_cost_gap']:.2%}，中位数 {statistics['greedy_median_relative_cost_gap']:.2%}。",
        f"按种子成簇的 bootstrap 描述性 95% 区间：平均差距 [{ci[0]:.2%}, {ci[1]:.2%}]；只有 {len(seed_means)} 个种子簇。", "",
        "二、精度扫描：每种规模/阈值/方法分别汇总"]
    for c in cells:
        if c["group"]=="precision": lines.append(f"N={c['vehicles']}, tol={c['tolerance']:g}, {c['method']}: {c['passes']}/{c['trials']} 通过，耗时中位数 {c['seconds_median']:.6f}s（包括失败）。")
    lines += ["", "三、缩放消融：关闭共同的行/列/目标缩放；PDHG 对角步长保持"]
    for c in cells:
        if c["group"]=="scaling": lines.append(f"N={c['vehicles']}, scaling={c['scaling']}, {c['method']}: {c['passes']}/{c['trials']} 通过，耗时中位数 {c['seconds_median']:.6f}s。")
    lines += ["", "四、PDHG 迭代预算（每配置 5 种子、每种子 1 次）"]
    for c in cells:
        if c["group"]=="budget": lines.append(f"N={c['vehicles']}, 上限={c['max_iterations']}: {c['passes']}/{c['trials']} 通过；证书分数中位数 {c['certificate_score_median']:.3g}；耗时中位数 {c['seconds_median']:.6f}s。")
    lines += ["", "五、对偶敏感性：单独增加某一台 RSU 的某一类容量"]
    for f in config["sensitivity_perturbations"]:
        sub=[r for r in sensitivity if r["fraction"]==f]
        errs=[r["prediction_error_over_baseline"] for r in sub]
        exact=sum(r["absolute_prediction_error"] <= 1e-6 for r in sub)
        lines.append(f"增加 {f:.1%}，{len(sub)} 个扰动；线性预测绝对误差 <=1e-6 的有 {exact} 个；预测误差/基线成本的中位数 {np.median(errs):.3g}，最大值 {max(errs):.3g}。")
    lines += ["", "六、结果可信度与解释边界",
        f"独立复核通过：{audit['verified_timed_solves']} 次算法比较、{audit['verified_reference_solves']} 次参考求解、{audit['verified_sensitivity_solves']} 次敏感性求解；从保存的原始变量与对偶重算约束、证书、目标和标签。",
        "耗时先对每实例重复取中位数，再跨种子取中位数/IQR；失败时间保留。HiGHS 内部精度固定为 1e-9。",
        "教学实现含迭代监控开销，HiGHS 使用预处理与成熟实现；时间差不能只归因于算法理论。",
        "目标是成本加未满足惩罚，并非单独最大化服务率。容量或连接增加后，最优成本的变化比服务率更有直接的优化解释。",
        "PDHG 达到上限表示该实现未在给定预算达到阈值，不表示实例不可解，也不表示所有一阶算法都有相同性能。",
        "缩放消融只检验共同尺度变换，不代表去除了算法内部所有预条件或预处理。",
        "只有 5 个随机种子，bootstrap 区间是描述性的；合成分布、可拆分工作量、固定单窗口限制外推范围。",
        "未测内存，未使用真实车辆轨迹，未进行协议/时延仿真，未验证工业规模。", "",
        "文件：plan.txt/config.json/jobs.json 为预先固定设计；runs.csv 为逐次记录；aggregate.csv 为每实例重复汇总；cell_summary.csv 为跨种子统计；",
        "sensitivity.csv/statistics.json 为敏感性与区间；audit.json 为复核；figures/ 含 6 组 PNG/SVG 图；source_snapshot/ 为运行源码。"]
    (output/"summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    figures(output,config,rows,instances,sensitivity)
    print("\n".join(lines[:15]))
    print(f"Analysis and 6 figure pairs saved to {output}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,default=BASE/"experiments/formal")
    analyze(parser.parse_args().experiment)
