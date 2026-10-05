"""Paired per-instance comparison of the integer-cost (non-degenerate schedule) runs."""
import json, math
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

here = Path(__file__).parent
PAIRS = [("fowler", "chordal_fowler", "chordal sparsification, Fowler parents"),
         ("full_order_local", "chordal_local", "chordal sparsification, local parents"),
         ("fowler", "chordal_local", "Fowler baseline vs best chordal")]


def per_instance(rows):
    out = {}
    for r in rows:
        t = r["seconds"] / r["reads"]
        p = r["optimal_reads"] / r["reads"]
        tts = t if p >= 1 else (math.inf if p == 0 else t * math.log(0.01) / math.log(1 - p))
        out[(r["instance"], r["model"])] = dict(feas=r["feasible_reads"] / r["reads"], opt=p, tts=tts,
            sec=r["seconds"], n=int(r["instance"].split("_")[1]))
    return out


def boot_ci(d, rng=np.random.default_rng(0)):
    means = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return np.percentile(means, [2.5, 97.5])


def report(name, subset=None):
    rows = json.loads((here / name).read_text())["records"]
    data = per_instance(rows)
    insts = sorted({i for i, _ in data})
    if subset:
        insts = [i for i in insts if subset(data[(i, rows[0]["model"])]["n"])]
    print(f"\n### {name}  ({len(insts)} instances)")
    for a, b, label in PAIRS:
        if (insts[0], a) not in data or (insts[0], b) not in data:
            continue
        print(f"-- {label}: {b} minus {a}")
        for metric in ("feas", "opt"):
            d = np.array([data[(i, b)][metric] - data[(i, a)][metric] for i in insts])
            lo, hi = boot_ci(d)
            nz = d[d != 0]
            p = wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
            print(f"   {metric:4}: mean diff {100*d.mean():+5.1f} pts  95% CI [{100*lo:+5.1f}, {100*hi:+5.1f}]  "
                  f"wins/losses/ties {int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}  Wilcoxon p={p:.3g}")
        ta = np.array([data[(i, a)]["tts"] for i in insts]); tb = np.array([data[(i, b)]["tts"] for i in insts])
        both = np.isfinite(ta) & np.isfinite(tb)
        ratio = np.exp(np.median(np.log(tb[both] / ta[both])))
        print(f"   TTS99 (wall clock): median ratio {b}/{a} = {ratio:.2f} over {both.sum()} instances both solved; "
              f"only {a} solved: {int((np.isfinite(ta)&~np.isfinite(tb)).sum())}, only {b} solved: {int((~np.isfinite(ta)&np.isfinite(tb)).sum())}")
        sa = sum(data[(i, a)]["sec"] for i in insts); sb = sum(data[(i, b)]["sec"] for i in insts)
        print(f"   total sampling time: {a} {sa:.1f}s, {b} {sb:.1f}s")


if __name__ == "__main__":
    report("integer_extended_results.json")
    for n in (8, 12, 16):
        report("integer_extended_results.json", subset=lambda m, n=n: m == n)
    report("integer_larger_results.json")
