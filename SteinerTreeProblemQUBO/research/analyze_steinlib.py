"""Per-instance table for the SteinLib runs: feasibility and best/median tree gap."""
import json
import sys
from pathlib import Path

here = Path(__file__).parent
MODELS = ("fowler", "full_order_local", "chordal_fowler", "chordal_local")


def main():
    records = []
    for name in sys.argv[1:] or ["steinlib_results.json", "steinlib_large_results.json"]:
        path = here / name
        if path.exists():
            records += json.loads(path.read_text())["records"]
    index = {(r["instance"], r["model"], r["solver"]): r for r in records}
    for solver in dict.fromkeys(r["solver"] for r in records):
        print(f"\n== {solver}: valid-tree % / best-tree gap % / median valid-tree gap %")
        print(f"{'inst':5} {'n':>3} {'m':>3} {'|T|':>3} {'size':>5} " + " ".join(f"{m:>22}" for m in MODELS))
        wins = dict(feas=0, best=0, n=0)
        for inst in sorted({r["instance"] for r in records}):
            rows = [index.get((inst, m, solver)) for m in MODELS]
            if None in rows:
                continue
            r0 = rows[0]
            cells = []
            for r in rows:
                best = "none" if r["best_feasible_cost"] is None else f"{100 * (r['best_feasible_cost'] / r['optimum'] - 1):.0f}"
                med = "-" if r["median_feasible_cost"] is None else f"{100 * (r['median_feasible_cost'] / r['optimum'] - 1):.0f}"
                cells.append(f"{100 * r['feasible_reads'] / r['reads']:3.0f}/{best:>4}/{med:>4}")
            print(f"{inst:5} {r0['n']:3} {r0['edges']:3} {r0['terminals']:3} {r0['chordal_edges'] / r0['complete_edges']:5.2f} "
                  + " ".join(f"{c:>22}" for c in cells))
            f, c = rows[0], rows[3]
            wins["n"] += 1
            wins["feas"] += c["feasible_reads"] > f["feasible_reads"]
            fb = f["best_feasible_cost"] or float("inf")
            cb = c["best_feasible_cost"] or float("inf")
            wins["best"] += (cb < fb) - (cb > fb) * 0
        print(f"chordal_local vs fowler: more valid trees on {wins['feas']}/{wins['n']}; strictly better best tree on {wins['best']}/{wins['n']}")


if __name__ == "__main__":
    main()
