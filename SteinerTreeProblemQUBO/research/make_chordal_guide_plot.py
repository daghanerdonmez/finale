"""Create the complexity-ratio figure used by chordal_formulation_guide.tex."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).with_name("general_results.json")
OUT = ROOT / "output" / "pdf"

FAMILIES = ["sparse", "grid", "geo_knn4", "er30", "er60"]
LABELS = {
    "sparse": "Sparse",
    "grid": "Grid",
    "geo_knn4": "Geometric kNN",
    "er30": "ER, p=0.3",
    "er60": "ER, p=0.6",
}
COLORS = ["#176B87", "#2D8C73", "#7B5EA7", "#D58C32", "#B84A4A"]


def main():
    data = json.loads(DATA.read_text())
    records = [r for r in data["records"] if r["solver"] == "neal"
               and r["model"] in {"fowler", "chordal_fowler"}]
    index = {(r["instance"], r["model"]): r for r in records}
    instances = {i["id"]: i for i in data["instances"]}
    sizes = sorted({i["n"] for i in instances.values()})

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.0), sharex=True, sharey=True)
    metrics = [("variables", "Binary-variable ratio"),
               ("interactions", "Quadratic-interaction ratio")]
    medians_at_20 = {}

    for ax, (metric, title) in zip(axes, metrics):
        for family, color in zip(FAMILIES, COLORS):
            values = []
            for n in sizes:
                ids = [key for key, inst in instances.items()
                       if inst["family"] == family and inst["n"] == n]
                ratios = [index[(key, "chordal_fowler")][metric]
                          / index[(key, "fowler")][metric] for key in ids]
                values.append(float(np.median(ratios)))
                if n == 20:
                    medians_at_20[(family, metric)] = (
                        float(np.median([index[(key, "fowler")][metric] for key in ids])),
                        float(np.median([index[(key, "chordal_fowler")][metric] for key in ids])),
                    )
            ax.plot(sizes, values, marker="o", linewidth=1.8, markersize=4.5,
                    color=color, label=LABELS[family])
        ax.axhline(1, color="#777777", linestyle="--", linewidth=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(sizes)
        ax.set_xlabel("Number of vertices, $n$")
        ax.set_ylim(0, 1.06)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Chordal / Fowler")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=5, columnspacing=1.0,
               handlelength=1.6)
    fig.suptitle("Model-size reduction with identical Fowler parent penalties", fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94), w_pad=2.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "chordal_complexity_ratios.pdf", bbox_inches="tight")
    fig.savefig(OUT / "chordal_complexity_ratios.png", dpi=220, bbox_inches="tight")

    print("Median Fowler -> chordal/Fowler counts at n=20")
    for family in FAMILIES:
        v = medians_at_20[(family, "variables")]
        q = medians_at_20[(family, "interactions")]
        print(f"{LABELS[family]:14s} vars {v[0]:.0f}->{v[1]:.0f}; "
              f"quadratic {q[0]:.0f}->{q[1]:.0f}")


if __name__ == "__main__":
    main()
