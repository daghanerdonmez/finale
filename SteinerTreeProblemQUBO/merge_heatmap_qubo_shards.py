"""
Merge multiple `heatmap_qubo_*.json` shard files produced by parallel runs of
`heatmap_benchmark_segment2.py` (with SEED_RANGE set on each worker) into a
single canonical file with a re-built per-combo summary.

The merger:
  1. Unions every shard's `instances` dict (same key = same (p, n, k, seed),
     so there should be no conflicts in a well-partitioned run).
  2. Cross-checks that incompatible meta fields (BQM version, num_sweeps,
     gurobi_source, etc.) agree between shards and warns otherwise.
  3. For each `(p, n, k)` combo, collects all non-discarded seeds, sorts them
     ascending by seed ID, takes the first `target_per_combo` as the canonical
     benchmark set, and rebuilds `summary` from those.
  4. Warns about any combo that is short of the target quota.

Typical usage from the project root:

    python -m SteinerTreeProblemQUBO.merge_heatmap_qubo_shards \\
        SteinerTreeProblemQUBO/logs/heatmap_qubo_*_seeds*.json \\
        -o SteinerTreeProblemQUBO/logs/heatmap_qubo_merged.json

You can also pass individual paths instead of a glob.
"""
import argparse
import glob
import json
import os
from datetime import datetime


# Meta fields that must agree between shards (otherwise results aren't
# comparable).  Mismatches produce a warning but don't abort.
CRITICAL_META_KEYS = (
    "gurobi_source",
    "bqm_version",
    "constraint_weight",
    "num_reads_per_trial",
    "num_trials",
    "stop_on_first_hit",
    "sqa_num_sweeps",
    "sqa_trotter",
    "max_weight",
    "node_count_list",
    "terminal_count_list",
    "edge_probability_list",
    "num_instances_per_combo",
)


def _load_shards(paths):
    shards = []
    for path in paths:
        with open(path) as f:
            shards.append((path, json.load(f)))
    return shards


def _check_meta_compat(shards):
    ref_path, ref_data = shards[0]
    ref_meta = ref_data["_meta"]
    for path, data in shards[1:]:
        m = data.get("_meta", {})
        for key in CRITICAL_META_KEYS:
            if m.get(key) != ref_meta.get(key):
                print(
                    f"[WARN] {os.path.basename(path)}: _meta.{key} = "
                    f"{m.get(key)!r} but {os.path.basename(ref_path)} has "
                    f"{ref_meta.get(key)!r}"
                )
    return ref_meta


def _union_instances(shards):
    """Return (instances dict, list of conflicting keys)."""
    merged = {}
    sources = {}
    conflicts = []
    for path, data in shards:
        for k, v in data.get("instances", {}).items():
            if k not in merged:
                merged[k] = v
                sources[k] = [path]
                continue
            sources[k].append(path)
            existing = merged[k]
            # On overlap, prefer a non-discarded record over a discarded one.
            if existing.get("discarded") and not v.get("discarded"):
                merged[k] = v
            # If both non-discarded and disagree on key metrics, flag it.
            if (
                not existing.get("discarded")
                and not v.get("discarded")
                and (
                    existing.get("solved") != v.get("solved")
                    or existing.get("first_hit_reads") != v.get("first_hit_reads")
                )
            ):
                conflicts.append((k, sources[k]))
    return merged, conflicts


def _rebuild_summary(instances, target_per_combo):
    combos = {}  # (p, n, k) -> list of (seed, inst)
    for inst in instances.values():
        key = (
            inst["edge_probability"],
            inst["node_count"],
            inst["terminal_count"],
        )
        combos.setdefault(key, []).append((inst["seed"], inst))

    summary = {}
    short_combos = []
    for (p, n, tk), items in sorted(combos.items()):
        items.sort(key=lambda x: x[0])

        valid = [
            (s, i)
            for s, i in items
            if not (i.get("discarded") or i.get("buggy_below_optimal"))
        ]
        discarded_ids = [
            s
            for s, i in items
            if i.get("discarded") or i.get("buggy_below_optimal")
        ]
        used = valid[:target_per_combo]
        leftover_ids = [s for s, _ in valid[target_per_combo:]]

        first_hits = [i.get("first_hit_reads") for _, i in used]
        solved_flags = [i.get("solved", False) for _, i in used]
        used_seed_ids = [s for s, _ in used]
        n_done = len(used)
        solved_count = sum(solved_flags)
        success_rate = (solved_count / n_done) if n_done else None
        solved_hits = [h for h in first_hits if h is not None]
        avg_first_hit = (
            sum(solved_hits) / len(solved_hits) if solved_hits else None
        )

        summary[f"p={p}|n={n}|k={tk}"] = {
            "edge_probability": p,
            "node_count": n,
            "terminal_count": tk,
            "num_instances_run": n_done,
            "num_instances_solved": solved_count,
            "success_rate": success_rate,
            "first_hits": first_hits,
            "avg_first_hit_reads_when_solved": avg_first_hit,
            "used_seed_ids": used_seed_ids,
            "discarded_seed_ids": discarded_ids,
            "leftover_seed_ids": leftover_ids,
            "reached_target": n_done >= target_per_combo,
        }
        if n_done < target_per_combo:
            short_combos.append((p, n, tk, n_done, len(discarded_ids)))

    return summary, short_combos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shards",
        nargs="+",
        help="shard JSON paths (glob patterns are expanded, e.g. "
        "'logs/heatmap_qubo_*_seeds*.json')",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="path to write the merged JSON"
    )
    parser.add_argument(
        "--target-per-combo",
        type=int,
        default=None,
        help="quota of valid seeds per (p, n, k) combo "
        "(default: _meta.num_instances_per_combo of the first shard)",
    )
    args = parser.parse_args()

    paths = []
    for pat in args.shards:
        matches = sorted(glob.glob(pat))
        paths.extend(matches if matches else [pat])
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no shard files matched")

    print(f"Merging {len(paths)} shard(s):")
    for p in paths:
        print(f"  - {p}")

    shards = _load_shards(paths)
    ref_meta = _check_meta_compat(shards)
    target = args.target_per_combo or ref_meta.get("num_instances_per_combo", 10)

    merged_instances, conflicts = _union_instances(shards)
    print(f"Union: {len(merged_instances)} unique instance keys")
    if conflicts:
        print(f"[WARN] {len(conflicts)} instance key(s) had overlapping, "
              f"non-matching results across shards:")
        for k, srcs in conflicts[:10]:
            print(f"  {k}  seen in {[os.path.basename(s) for s in srcs]}")
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more")

    summary, short_combos = _rebuild_summary(merged_instances, target)
    print(f"Rebuilt summary for {len(summary)} combo(s), target={target}")
    if short_combos:
        print(
            f"[WARN] {len(short_combos)} combo(s) short of target; run more "
            f"seeds to top them up:"
        )
        for p, n, tk, got, disc in short_combos:
            print(
                f"  p={p} n={n} k={tk}: {got}/{target} valid "
                f"(discarded {disc})"
            )

    out = {
        "_meta": {
            **ref_meta,
            "merged_from": [os.path.basename(p) for p in paths],
            "merged_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "target_per_combo": target,
            # Scrub shard-specific fields that don't apply to the merged doc.
            "seed_range": None,
        },
        "instances": merged_instances,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
