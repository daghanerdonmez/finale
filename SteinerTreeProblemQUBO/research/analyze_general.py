"""Summarise general_results.json: paired per-instance comparisons by solver and family."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

here = Path(__file__).parent
MODELS = ("fowler", "full_order_local", "chordal_fowler", "chordal_local")
PAIRS = [
    ("fowler", "full_order_local"),       # local coefficient only, full ordering
    ("chordal_fowler", "chordal_local"), # local coefficient only, chordal ordering
    ("fowler", "chordal_fowler"),        # chordal sparsification only, Fowler parents
    ("full_order_local", "chordal_local"), # chordal sparsification only, local parents
    ("fowler", "chordal_local"),         # both changes together
]


def tts99(r):
    t = r["seconds"] / r["reads"]
    p = r["optimal_reads"] / r["reads"]
    return t if p >= 1 else (math.inf if p == 0 else t * math.log(0.01) / math.log(1 - p))


def paired(index, insts, solver, a, b, metric):
    d = np.array([index[(i, b, solver)][metric] - index[(i, a, solver)][metric] for i in insts])
    rng = np.random.default_rng(0)
    lo, hi = np.percentile([rng.choice(d, len(d)).mean() for _ in range(4000)], [2.5, 97.5])
    nz = d[d != 0]
    p = wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
    return dict(mean=d.mean(), lo=lo, hi=hi, wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                ties=int((d == 0).sum()), p=p)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "general_results.json"
    data = json.loads((here / name).read_text())
    instances = {i["id"]: i for i in data["instances"]}
    index = {}
    for r in data["records"]:
        r = dict(r, feas=r["feasible_reads"] / r["reads"], opt=r["optimal_reads"] / r["reads"], tts=tts99(r),
                 # Best-tree gap to optimum; no valid tree at all is scored as a 100% gap.
                 gap=1.0 if r["best_feasible_cost"] is None else r["best_feasible_cost"] / r["optimum"] - 1)
        index[(r["instance"], r["model"], r["solver"])] = r
    solvers = list(dict.fromkeys(r["solver"] for r in data["records"]))
    insts = sorted(instances)
    degenerate = sum(r["beta_range"][1] > 1e6 for r in data["records"] if r["beta_range"])
    print(f"{len(insts)} instances; degenerate Neal schedules: {degenerate}")
    nontrivial = [i for i in insts if instances[i]["feasible_upper"] > instances[i]["optimum"]]
    print(f"instances where pruned MST is not already optimal: {len(nontrivial)}")
    summary = {}

    for solver in solvers:
        print(f"\n=================== {solver} ===================")
        print("overall mean rates:  " + "  ".join(
            f"{m}: feas {100*np.mean([index[(i,m,solver)]['feas'] for i in insts]):.1f}% "
            f"opt {100*np.mean([index[(i,m,solver)]['opt'] for i in insts]):.1f}%" for m in MODELS))
        for a, b in PAIRS:
            print(f"-- {b} minus {a}")
            for metric in ("feas", "opt", "gap"):
                s = paired(index, insts, solver, a, b, metric)
                summary[f"{solver}|{b}-{a}|{metric}"] = s
                better, worse = (s['losses'], s['wins']) if metric == "gap" else (s['wins'], s['losses'])
                print(f"   {metric}: {100*s['mean']:+5.1f} pts [{100*s['lo']:+5.1f}, {100*s['hi']:+5.1f}]  "
                      f"{b} better/worse/tie {better}/{worse}/{s['ties']}  p={s['p']:.2g}")
            ta = np.array([index[(i, a, solver)]["tts"] for i in insts])
            tb = np.array([index[(i, b, solver)]["tts"] for i in insts])
            both = np.isfinite(ta) & np.isfinite(tb)
            ratio = float(np.exp(np.median(np.log(tb[both] / ta[both])))) if both.any() else float("nan")
            print(f"   TTS99 median ratio {ratio:.2f} ({both.sum()} both solved); only {a}: "
                  f"{int((np.isfinite(ta) & ~np.isfinite(tb)).sum())}, only {b}: {int((~np.isfinite(ta) & np.isfinite(tb)).sum())}")

        print("\n   by family (chordal_local minus fowler; size = mean chordal/complete ordering edges):")
        for fam in dict.fromkeys(instances[i]["family"] for i in insts):
            sub = [i for i in insts if instances[i]["family"] == fam]
            size = np.mean([instances[i]["chordal_edges"] / instances[i]["complete_edges"] for i in sub])
            f = paired(index, sub, solver, "fowler", "chordal_local", "feas")
            o = paired(index, sub, solver, "fowler", "chordal_local", "opt")
            g = paired(index, sub, solver, "fowler", "chordal_local", "gap")
            fr = np.mean([index[(i, "fowler", solver)]["opt"] for i in sub])
            print(f"   {fam:9} size {size:.2f}  feas {100*f['mean']:+5.1f} (W/L {f['wins']}/{f['losses']})  "
                  f"opt {100*o['mean']:+5.1f} (W/L {o['wins']}/{o['losses']}, p={o['p']:.2g})  fowler opt {100*fr:.1f}%  best-gap {100*g['mean']:+.1f} pts (W/L {g['losses']}/{g['wins']})")
        print("   by size n (chordal_local minus fowler):")
        for n in sorted({instances[i]["n"] for i in insts}):
            sub = [i for i in insts if instances[i]["n"] == n]
            f = paired(index, sub, solver, "fowler", "chordal_local", "feas")
            o = paired(index, sub, solver, "fowler", "chordal_local", "opt")
            print(f"   n={n:2}  feas {100*f['mean']:+5.1f} (W/L {f['wins']}/{f['losses']})  opt {100*o['mean']:+5.1f} (W/L {o['wins']}/{o['losses']}, p={o['p']:.2g})")
        print("   by terminal fraction:")
        for t in sorted({instances[i]["terminal_fraction"] for i in insts}):
            sub = [i for i in insts if instances[i]["terminal_fraction"] == t]
            f = paired(index, sub, solver, "fowler", "chordal_local", "feas")
            o = paired(index, sub, solver, "fowler", "chordal_local", "opt")
            print(f"   t={t:.2f}  feas {100*f['mean']:+5.1f}  opt {100*o['mean']:+5.1f} (W/L {o['wins']}/{o['losses']}, p={o['p']:.2g})")
        ratio = [instances[i]["chordal_edges"] / instances[i]["complete_edges"] for i in insts]
        adv = [index[(i, "chordal_local", solver)]["opt"] - index[(i, "fowler", solver)]["opt"] for i in insts]
        advf = [index[(i, "chordal_local", solver)]["feas"] - index[(i, "fowler", solver)]["feas"] for i in insts]
        print(f"   Spearman(size ratio, advantage): feas rho={spearmanr(ratio, advf)[0]:+.2f}, opt rho={spearmanr(ratio, adv)[0]:+.2f}")
    (here / name.replace("results", "summary")).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
