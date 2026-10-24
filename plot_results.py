"""Generate standalone figures from saved benchmark data; never rerun solvers."""
import argparse
import csv
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE.parents[1] / "work/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"highs": "HiGHS reference", "simplex": "Primal simplex", "ipm": "Predictor-corrector IPM",
         "pdhg": "Diagonal PDHG", "greedy": "Greedy baseline"}
COLORS = {"highs": "#333333", "simplex": "#2676B8", "ipm": "#C55A11", "pdhg": "#008878", "greedy": "#A45AA0"}


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(output / (name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(experiment):
    with (experiment / "aggregate.csv").open() as f:
        aggregate = list(csv.DictReader(f))
    with (experiment / "runs.csv").open() as f:
        runs = list(csv.DictReader(f))
    manifest = json.loads((experiment / "manifest.json").read_text())
    methods = manifest["config"]["methods"]
    tolerance = manifest["config"]["tolerance"]
    output = experiment / "figures"; output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    ax = axes[0]
    sizes = sorted({int(r["vehicles"]) for r in aggregate if "scale" in r["groups"].split("|")})
    for method in methods:
        med, lower, upper = [], [], []
        for n in sizes:
            values = [float(r["seconds_median_all_runs"]) for r in aggregate
                      if r["method"] == method and int(r["vehicles"]) == n
                      and "scale" in r["groups"].split("|") and r["seconds_median_all_runs"]]
            if not values: raise ValueError(f"No runtime observations for {method}, N={n}")
            center = float(np.median(values)); med.append(center)
            lower.append(center - np.quantile(values, 0.25)); upper.append(np.quantile(values, 0.75) - center)
        ax.errorbar(sizes, med, yerr=[lower, upper], marker="o", markersize=4,
                    capsize=3, color=COLORS[method], label=NAMES[method], linewidth=1.7)
    ax.set_yscale("log"); ax.set_xticks(sizes)
    ax.set_xlabel("Vehicles (RSUs: 3, 4, 6, 8)"); ax.set_ylabel("Solver time (seconds, log scale)")
    ax.set_title("A  Runtime versus problem size")
    ax.grid(axis="y", alpha=0.2); ax.legend(fontsize=8)
    ax = axes[1]
    rates, counts = [], []
    for method in methods:
        selected = [r for r in runs if r["method"] == method]
        passed = sum(r["passed"] == "True" for r in selected)
        rates.append(passed / len(selected)); counts.append(f"{passed}/{len(selected)}")
    bars = ax.bar(range(len(methods)), rates, color=[COLORS[m] for m in methods], width=0.62)
    ax.set_xticks(range(len(methods)), ["HiGHS", "Simplex", "IPM", "PDHG", "Greedy*"], rotation=15)
    ax.set_ylim(0, 1.17); ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    for bar, count in zip(bars, counts): ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .025, count, ha="center", fontsize=9)
    ax.set_title("B  Validation pass rate"); ax.set_ylabel("Fraction of timed solves")
    config = manifest["config"]
    fig.suptitle(f"Paired local benchmark: {manifest['instance_count']} instances, {config['repetitions']} repetitions, tolerance {tolerance:g}", fontsize=12)
    fig.supxlabel(f"Runtime includes iteration-limit runs; median and IQR across {len(config['seeds'])} seeds.  *Greedy: feasibility only.", fontsize=9)
    save(fig, output, "runtime_and_validation")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for ax, group, field, title in zip(axes, ["compute", "upload"], ["rho_compute", "rho_upload"],
                                     ["A  Compute capacity", "B  Upload capacity"]):
        ratios = sorted({float(r[field]) for r in aggregate if group in r["groups"].split("|")})
        for method in ["highs", "greedy"]:
            med, lo, hi = [], [], []
            for ratio in ratios:
                values = [float(r["satisfaction_median_passed"]) for r in aggregate
                          if method == r["method"] and group in r["groups"].split("|")
                          and float(r[field]) == ratio and r["satisfaction_median_passed"]]
                center = float(np.median(values)); med.append(center)
                lo.append(center - np.quantile(values, .25)); hi.append(np.quantile(values, .75) - center)
            ax.errorbar(ratios, med, yerr=[lo, hi], marker="o", color=COLORS[method],
                        label=NAMES[method], capsize=4, linewidth=1.8)
        ax.set_xticks(ratios); ax.set_ylim(0.35, 1.03); ax.set_title(title)
        ax.set_xlabel("Capacity / aggregate resource demand"); ax.set_ylabel("Served workload / total workload")
        ax.grid(axis="y", alpha=.2); ax.legend(fontsize=9)
    fig.suptitle("Service under resource limits: 30 vehicles, 4 RSUs, other resource ratio = 1.4", fontsize=11)
    fig.supxlabel("Median and IQR across 3 paired seeds. LP minimizes cost plus unserved penalty, not service alone.", fontsize=9)
    save(fig, output, "capacity_and_service")

    scale = [r for r in aggregate if "scale" in r["groups"].split("|")]
    nmax = max(int(r["vehicles"]) for r in scale)
    chosen = sorted({r["instance"] for r in scale if int(r["vehicles"]) == nmax})[0]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for method in ["simplex", "ipm", "pdhg"]:
        trace = json.loads((experiment / "traces" / f"{chosen}__{method}__r0.json").read_text())
        history = trace["history"]
        ax.plot([r["seconds"] for r in history], [max(1e-16, r["certificate_score"]) for r in history],
                color=COLORS[method], label=NAMES[method], linewidth=1.8)
    ax.axhline(tolerance, color="#777777", linestyle="--", linewidth=1, label="Stopping tolerance")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Elapsed solver time (seconds, log scale)")
    ax.set_ylabel("Maximum normalized residual / gap")
    ax.set_title(f"Convergence trace: {chosen}", fontsize=11)
    ax.grid(alpha=.2); ax.legend(fontsize=9)
    save(fig, output, "convergence")
    print(f"Saved figures to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=BASE / "experiments/batch")
    plot(parser.parse_args().experiment)
