"""Create a research figure and summary tables from the saved pilot data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .benchmark_repairs import make_instance, big_m, stats
from .steiner_to_bqm_chordal import steiner_to_bqm_chordal
from .steiner_to_bqm_binary_order import steiner_to_bqm_binary_order
from SteinerTreeProblemQUBO.AlexFowler.steiner_to_bqm_alex import steiner_to_bqm_ordering


def summarize(rows):
    result = []
    for model in dict.fromkeys(r["model"] for r in rows):
        subset = [r for r in rows if r["model"] == model]
        total = sum(r["reads"] for r in subset)
        result.append(dict(model=model, instances=len(subset),
            solved=sum(r["optimal_reads"] > 0 for r in subset),
            feasible_percent=100 * sum(r["feasible_reads"] for r in subset)/total,
            optimal_percent=100 * sum(r["optimal_reads"] for r in subset)/total,
            zero_penalty_percent=100 * sum(r["zero_penalty_reads"] for r in subset)/total))
    return result


def main():
    here = Path(__file__).parent
    small = []
    for name in ("pilot_results.json", "refinement_results.json"):
        small.extend(json.loads((here / name).read_text())["records"])
    larger = json.loads((here / "larger_results.json").read_text())["records"]
    summaries = dict(small=summarize(small), larger=summarize(larger))
    for group, rows in summaries.items():
        print(group)
        for row in rows:
            print("{model:18} {solved:2}/{instances:2} feas={feasible_percent:5.1f}% "
                  "opt={optimal_percent:5.1f}% zero={zero_penalty_percent:5.1f}%".format(**row))
    (here / "summary.json").write_text(json.dumps(summaries, indent=2))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    names = ["big_m", "circuit", "binary_order", "domain_wall", "fowler", "chordal_local"]
    labels = ["Big-M depths", "Bitwise depths", "Shared comparator", "Domain wall", "Full ordering", "Chordal + local parents"]
    selected = {r["model"]: r for r in summaries["small"]}
    y = np.arange(len(names))
    axes[0].barh(y + .18, [selected[m]["feasible_percent"] for m in names], height=.32,
                 color="#5185a6", label="Feasible tree")
    axes[0].barh(y - .18, [selected[m]["optimal_percent"] for m in names], height=.32,
                 color="#194d40", label="Optimal tree")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Samples (%)")
    axes[0].set_title("A. Small-instance pilot")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)

    builders = {"Big-M depths": big_m, "Full ordering": steiner_to_bqm_ordering,
                "Shared comparator": steiner_to_bqm_binary_order,
                "Chordal + local parents": lambda p, P: steiner_to_bqm_chordal(p, P, parent_encoding="local")}
    colors = ["#ad534a", "#be912b", "#5185a6", "#194d40"]
    sizes = [8, 16, 32, 64]
    scaling = []
    for (label, builder), color in zip(builders.items(), colors):
        counts, coeff = [], []
        for n in sizes:
            problem = make_instance("ladder", n, 0)
            bqm = builder(problem, 1.0)
            row = dict(model=label, n=n, **stats(bqm, 1.0))
            scaling.append(row)
            counts.append(row["interactions"])
            coeff.append(row["max_abs_qubo_quadratic"])
        axes[1].plot(sizes, counts, "o-", color=color, label=label, linewidth=1.8)
        axes[2].plot(sizes, coeff, marker="o", color=color, label=label, linewidth=1.8,
                     linestyle="--" if label == "Chordal + local parents" else "-")
    (here / "scaling_results.json").write_text(json.dumps(scaling, indent=2))
    for ax in axes[1:]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes, sizes)
        ax.set_xlabel("Vertices in ladder graph")
        ax.grid(axis="y", alpha=.15)
    axes[1].set_ylabel("Nonzero quadratic interactions")
    axes[1].set_title("B. Encoding size")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    axes[2].set_ylabel("Maximum |Qᵢⱼ| / penalty")
    axes[2].set_title("C. Quadratic coefficient scale")
    fig.suptitle("Steiner-tree QUBOs: precision repair helps; sparse ordering is the stronger pilot direction", fontsize=13)
    fig.text(.02, .025, "A: 18 instances, n = 8/12/16; 100 reads × 2,000 sweeps; Neal SA; safe common penalty > feasible-tree upper bound.\n"
             "B–C: model counts only, not annealing outcomes. Shared comparator and chordal/local coefficient curves coincide.", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .12, 1, .94), w_pad=2.2)
    fig.savefig(here / "comparison.png", dpi=180)


if __name__ == "__main__":
    main()
