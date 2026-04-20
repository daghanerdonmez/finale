"""Solve a saved Steiner Tree Problem instance with the Li et al. ILP.

Usage (from the project root, ``finale/``):

    python SteinerTreeProblemQUBO/Li_et_al/solve_instance.py <name>

<name> can be:
  * a bare instance name  (e.g. "demo")
  * a file name           (e.g. "demo.json")
  * an absolute/relative path to a JSON file

Bare names and file names are looked up inside
``SteinerTreeProblemQUBO/STPInstances/``, where ``instance_editor.py`` writes
its output.
"""

import argparse
import json
import sys
from pathlib import Path


_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
INSTANCES_DIR = PROJECT_ROOT / "SteinerTreeProblemQUBO" / "STPInstances"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.Li_et_al.gurobi_solver import solve_ilp_li_et_al


def resolve_instance_path(name: str) -> Path:
    p = Path(name)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(INSTANCES_DIR / name)
        if p.suffix.lower() != ".json":
            candidates.append(INSTANCES_DIR / f"{name}.json")
        candidates.append(Path.cwd() / name)

    for c in candidates:
        if c.is_file():
            return c

    lines = [f"Could not find instance '{name}'.", "Searched:"]
    lines.extend(f"  {c}" for c in candidates)
    if INSTANCES_DIR.is_dir():
        available = sorted(p.stem for p in INSTANCES_DIR.glob("*.json"))
        if available:
            lines.append("Available instances under STPInstances/:")
            lines.extend(f"  {a}" for a in available)
        else:
            lines.append(f"No *.json files found in {INSTANCES_DIR}.")
    raise FileNotFoundError("\n".join(lines))


def load_steiner_tree(path: Path) -> SteinerTree:
    with open(path) as f:
        data = json.load(f)
    try:
        nodes = [str(n) for n in data["nodes"]]
        edges = [(str(u), str(v), int(w)) for u, v, w in data["edges"]]
        terminals = [str(t) for t in data["terminals"]]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid instance file {path.name}: {e}") from e
    return SteinerTree(nodes, edges, terminals)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solve a saved Steiner Tree instance with the Li et al. ILP.",
    )
    parser.add_argument(
        "instance",
        help="Name (or path) of an instance under STPInstances/ (e.g. 'demo' or 'demo.json').",
    )
    parser.add_argument(
        "--non-edge-penalty", type=float, default=10000.0,
        help="Penalty weight for non-edges in the Li et al. formulation (default: %(default)s).",
    )
    args = parser.parse_args()

    try:
        path = resolve_instance_path(args.instance)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    try:
        problem = load_steiner_tree(path)
    except (ValueError, OSError) as e:
        print(f"Failed to load instance: {e}", file=sys.stderr)
        return 1

    print(f"Instance:  {path}")
    print(f"Nodes:     {len(problem.nodes)}")
    print(f"Edges:     {len(problem.edges)}")
    print(f"Terminals: {problem.terminals}")
    print()
    print("Running Gurobi ILP (Li et al. formulation)...")

    result = solve_ilp_li_et_al(problem, non_edge_penalty=args.non_edge_penalty)

    status = result["status"]
    print()
    print(f"Status:    {status}")
    if status != "OPTIMAL":
        print("Solver did not return an optimal solution.")
        return 2

    print(f"Objective: {result['objective']:.4f}")
    print(f"Tree cost: {result['tree_cost']:.4f}")
    print(f"Tree edges ({len(result['edges'])}):")
    for u, v, w in result["edges"]:
        print(f"  {u} -- {v}   (w={w})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
