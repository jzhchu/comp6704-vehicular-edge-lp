"""Recompute paper statistics and figures from the immutable formal batch only."""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE.parents[1]/"work/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_formal import MAIN_GROUPS, load_rows, per_instance, cell_summary
from benchmark import write_csv
from formal_suite import dump

SOURCE = BASE/"experiments/formal"
DEST = BASE/"experiments/report_analysis"


def run(source=SOURCE, destination=DEST):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = json.loads((source/"manifest.json").read_text())
    assert json.loads((source/"audit.json").read_text())["audit_passed"]
    checked = {}
    def read(relative):
        path = source/relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = manifest["result_sha256"].get(relative)
        if expected is None and relative.startswith("instances/"):
            expected = manifest["instance_sha256"][path.stem]
        if expected is not None: assert digest == expected, relative
        checked[relative] = digest
        return json.loads(path.read_text())

    rows = load_rows(source)
    assert hashlib.sha256((source/"runs.csv").read_bytes()).hexdigest() == manifest["result_sha256"]["runs.csv"]
    cells = cell_summary(per_instance(rows))
    main = [r for r in rows if r["group"] in MAIN_GROUPS]
    cases = sorted({r["instance"] for r in main})
    components = []
    def decompose(data, solution):
        x, u = np.array(solution["allocation"]), np.array(solution["unserved"])
        c = np.array([[0 if v is None else v for v in rr] for rr in data["cost"]])
        service, penalty = float(np.sum(x*c)), float(np.array(data["unserved_penalty"])@u)
        demand = float(np.sum(data["demand"]))
        np.testing.assert_allclose(service+penalty, solution["metrics"]["objective"], rtol=1e-12, atol=1e-9)
        return dict(service_cost=service, unserved_penalty=penalty, objective=service+penalty,
            total_demand=demand, service_cost_per_demand=service/demand,
            penalty_per_demand=penalty/demand, objective_per_demand=(service+penalty)/demand,
            satisfaction=float(np.sum(x)/demand))
    for name in cases:
        data = read(f"instances/{name}.json")
        meta = next(r for r in main if r["instance"]==name)
        reference = read(f"references/{name}.json")
        lp = decompose(data, reference)
        trials = [r for r in main if r["instance"]==name and r["method"]=="greedy"]
        greedy = []
        for row in trials:
            assert row["passed"]
            greedy.append(decompose(data, read(f"traces/{row['job_id']}.json")))
        for gg in greedy:
            np.testing.assert_allclose(list(gg.values()), list(greedy[0].values()), rtol=1e-12, atol=1e-9)
        for method, value in [("highs",lp),("greedy",greedy[0])]:
            components.append(dict(instance=name, group=meta["group"], vehicles=meta["vehicles"],
                rsus=meta["rsus"], seed=meta["seed"], rho_compute=meta["rho_compute"],
                rho_upload=meta["rho_upload"], radius=meta["radius"], method=method, **value))
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination/"cost_components.csv", components)
    averages = {method:{k:float(np.mean([r[k] for r in components if r["method"]==method]))
        for k in ["service_cost_per_demand","penalty_per_demand","objective_per_demand","satisfaction"]}
        for method in ["highs","greedy"]}
    difference = {k:averages["greedy"][k]-averages["highs"][k] for k in averages["highs"]}
    paired_gaps = [(next(r for r in components if r["instance"]==name and r["method"]=="greedy")["objective"]-
                    next(r for r in components if r["instance"]==name and r["method"]=="highs")["objective"])/
                    max(1.,abs(next(r for r in components if r["instance"]==name and r["method"]=="highs")["objective"]))
                    for name in cases]
    previous = json.loads((source/"statistics.json").read_text())
    np.testing.assert_allclose(np.mean(paired_gaps),previous["greedy_mean_relative_cost_gap"],rtol=1e-12)
    gap_by_case = dict(zip(cases, paired_gaps))
    seed_means = [np.mean([gap_by_case[r["instance"]] for r in components
                          if r["method"]=="highs" and r["seed"]==seed])
                  for seed in manifest["config"]["seeds"]]
    config = manifest["config"]
    bootstrap = np.mean(np.random.default_rng(config["bootstrap_seed"]).choice(
        seed_means, size=(config["bootstrap_samples"],len(seed_means)),replace=True),axis=1)
    np.testing.assert_allclose(np.quantile(bootstrap,[.025,.975]),
        previous["greedy_seed_cluster_bootstrap_95_percentile_interval"],rtol=1e-12)
    monotonic = []
    for group, field in [("compute","rho_compute"),("upload","rho_upload"),("connectivity","radius")]:
        for seed in manifest["config"]["seeds"]:
            selected = sorted([r for r in components if r["method"]=="highs" and group in r["group"].split("|") and r["seed"]==seed],key=lambda r:r[field])
            differences = np.diff([r["objective"] for r in selected])
            assert max(differences) <= 1e-7
            monotonic.append(dict(group=group, seed=seed, checked_steps=len(differences),
                maximum_objective_increase=float(max(differences))))
    write_csv(destination/"monotonicity_checks.csv",monotonic)
    # Use the smallest pre-specified seed at N=100. Same input, common scaling.
    chosen = f"n100_m8_c1_b1_r0.25_s{min(manifest['config']['seeds'])}"
    curves = []
    for method in ["simplex","ipm","pdhg"]:
        match = next(r for r in rows if r["instance"]==chosen and r["group"]=="scale" and r["method"]==method and r["repetition"]==0)
        curves.append(dict(label=method+" 100k" if method=="pdhg" else method,row=match,
                           trace=read(f"traces/{match['job_id']}.json")))
    match = next(r for r in rows if r["instance"]==chosen and r["group"]=="budget" and r["max_iterations"]==1000000)
    curves.append(dict(label="pdhg 1M",row=match,trace=read(f"traces/{match['job_id']}.json")))
    curve_rows = []
    for curve in curves:
        best = float("inf")
        for h in curve["trace"]["history"]:
            best = min(best,h["certificate_score"])
            curve_rows.append(dict(instance=chosen,label=curve["label"],job_id=curve["row"]["job_id"],
                iteration=h["iteration"],seconds=h["seconds"],certificate_score=h["certificate_score"],
                best_certificate_so_far=best))
    write_csv(destination/"convergence_curves.csv",curve_rows)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                        "axes.spines.right":False,"legend.frameon":False})
    fig,axes = plt.subplots(1,2,figsize=(7.1,2.7),layout="constrained")
    palette={"simplex":"#2676B8","ipm":"#C55A11","pdhg 100k":"#008878","pdhg 1M":"#777777"}
    names={"simplex":"Simplex","ipm":"IPM","pdhg 100k":"PDHG 100k","pdhg 1M":"PDHG 1M"}
    for curve in list(reversed(curves)):
        selected=[r for r in curve_rows if r["label"]==curve["label"]]
        for ax,field in zip(axes,["iteration","seconds"]):
            xs=[r[field]+1 if field=="iteration" else max(1e-6,r[field]) for r in selected]
            ys=[max(1e-16,r["best_certificate_so_far"]) for r in selected]
            ax.plot(xs,ys,label=names[curve["label"]],color=palette[curve["label"]],
                    linestyle="--" if curve["label"]=="pdhg 100k" else "-",linewidth=1.2)
            ax.scatter([xs[-1]],[ys[-1]],s=13,color=palette[curve["label"]])
    for ax in axes:
        ax.set_xscale("log");ax.set_yscale("log");ax.axhline(1e-5,color="black",ls=":",linewidth=.9)
        ax.grid(alpha=.2);ax.set_ylabel("Best certificate so far")
    axes[0].set_xlabel("Iteration + 1");axes[1].set_xlabel("Elapsed solver time (s)")
    axes[0].set_title("A  Iteration history");axes[1].set_title("B  Time history")
    axes[1].legend(fontsize=7,loc="upper right")
    for ext in ["png","svg"]:fig.savefig(destination/f"convergence.{ext}",dpi=260,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    # A compact three-panel figure with print-sized labels.
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.65),layout="constrained")
    endpoints=[]
    for ax,(group,field,title) in zip(axes,[("compute","rho_compute","Compute capacity"),
            ("upload","rho_upload","Upload capacity"),("connectivity","radius","Connection radius")]):
        selected=[r for r in components if group in r["group"].split("|")]
        xs=sorted({r[field] for r in selected})
        for method,color,label in [("highs","#333333","LP optimum"),("greedy","#A45AA0","Greedy")]:
            values=[[r["satisfaction"] for r in selected if r[field]==x and r["method"]==method] for x in xs]
            med=np.median(values,axis=1);lo=med-np.quantile(values,.25,axis=1);hi=np.quantile(values,.75,axis=1)-med
            ax.errorbar(xs,med,yerr=[lo,hi],marker="o",ms=3,capsize=2,color=color,label=label,linewidth=1.1)
        for x in [xs[0],xs[-1]]:
            record=dict(group=group,parameter=x)
            for method in ["highs","greedy"]:
                sub=[r for r in selected if r[field]==x and r["method"]==method]
                for metric in ["objective","satisfaction"]:record[f"{method}_{metric}_median"]=float(np.median([r[metric] for r in sub]))
            endpoints.append(record)
        ax.set_ylim(.25,1.04);ax.set_title(title);ax.set_xticks(xs);ax.grid(axis="y",alpha=.2)
        ax.set_xlabel("Radius" if group=="connectivity" else "Capacity ratio")
    axes[0].set_ylabel("Served workload / demand");axes[0].legend(fontsize=7,loc="lower right")
    for ext in ["png","svg"]:fig.savefig(destination/f"resource_service.{ext}",dpi=260,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    write_csv(destination/"resource_endpoints.csv",endpoints)
    info=dict(averages=averages,greedy_minus_lp=difference,
        penalty_share_of_normalized_objective_difference=difference["penalty_per_demand"]/difference["objective_per_demand"],
        cost_gap_statistics=previous, normalized_case_count=len(cases),
        all_counts={method:dict(passes=sum(r["passed"] for r in rows if r["method"]==method),
                                trials=sum(r["method"]==method for r in rows)) for method in manifest["config"]["methods"]},
        main_counts={method:dict(passes=sum(r["passed"] for r in main if r["method"]==method),
                                 trials=sum(r["method"]==method for r in main)) for method in manifest["config"]["methods"]},
        main_passed_max_reference_gap={method:max(r["reference_gap"] for r in main if r["method"]==method and r["passed"])
            for method in ["highs","simplex","ipm","pdhg"]},
        representative_instance=chosen,
        representative_curves=[dict(label=c["label"],job_id=c["row"]["job_id"],status=c["row"]["status"],
            iterations=c["row"]["iterations"],seconds=c["row"]["seconds"],
            final_certificate=c["trace"]["metrics"]["certificate_score"]) for c in curves],
        monotonicity=dict(paired_sequences=len(monotonic),checked_steps=sum(r["checked_steps"] for r in monotonic),passed=True),
        note="Post-processing only. Each of the 90 different main-experiment inputs has equal weight. Costs are divided by each input's total demand before averaging; relative cost gaps are averaged separately.")
    dump(destination/"paper_statistics.json",info)
    dump(destination/"cells.json",cells)
    sensitivity=list(csv.DictReader((source/"sensitivity.csv").open()))
    dump(destination/"sensitivity_summary.json",[{"fraction":f,"count":len(sub),
        "exact_to_1e_6":sum(float(r["absolute_prediction_error"])<=1e-6 for r in sub),
        "maximum_error_over_baseline":max(float(r["prediction_error_over_baseline"]) for r in sub)}
        for f in manifest["config"]["sensitivity_perturbations"]
        for sub in [[r for r in sensitivity if float(r["fraction"])==f]]])
    dump(destination/"analysis_manifest.json",dict(source_experiment=os.path.relpath(source, destination),solver_runs_added=0,
        source_files_sha256=checked,analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()
    run(args.experiment, args.output)
