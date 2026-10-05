# Quick four-model comparison

This experiment compares the two independent formulation choices:

| Model | Ordering penalty | Nonterminal parent coefficient |
|---|---|---|
| `fowler` | Complete order on all non-root vertices | Fowler's global coefficient \(n=|V|\) |
| `full_order_local` | Complete order on all non-root vertices | Local coefficient \(d_v^+ + 1\) |
| `chordal_fowler` | Order only on a chordal completion | Fowler's global coefficient \(n\) |
| `chordal_local` | Order only on a chordal completion | Local coefficient \(d_v^+ + 1\) |

The local coefficient changes QUBO weights but does not change the variables or coupler locations. The chordal construction reduces the ordering variables and couplers.

## Test setup

- 20 instances: sparse, grid, geometric 4-nearest-neighbour, ER(0.30), and ER(0.60)
- Sizes \(n\in\{12,16\}\), terminal fraction 25%, two seeds per family and size
- Integer edge costs
- Neal simulated annealing: 100 reads and 1,000 sweeps per model and instance
- Penalty multiplier \(P=\lceil1.05\,U\rceil\), where \(U\) is the benchmark's feasible upper bound
- All four models used the same instance and annealing settings in each paired comparison

## Mean results

| Model | Feasible samples | Optimal samples | Mean variables | Mean quadratic interactions |
|---|---:|---:|---:|---:|
| `fowler` | 86.4% | 17.8% | 139.8 | 1,372.3 |
| `full_order_local` | 87.7% | 17.6% | 139.8 | 1,372.3 |
| `chordal_fowler` | 91.1% | 16.3% | 95.2 | 601.3 |
| `chordal_local` | **93.0%** | **18.6%** | 95.2 | 601.3 |

No Neal run produced a degenerate automatically selected temperature schedule.

## Paired effects

Replacing Fowler's global coefficient by the local coefficient while retaining the complete order gave:

- Feasibility: +1.2 percentage points, bootstrap 95% CI \([-0.4,+3.1]\)
- Optimality: -0.1 points, bootstrap 95% CI \([-2.3,+2.0]\)
- Median TTS99 ratio: 0.84 among the 16 instances solved by both models

Thus this quick run does not show a clear advantage for the local coefficient by itself under complete ordering.

Replacing Fowler's coefficient by the local coefficient inside the chordal model gave:

- Feasibility: +2.0 points, bootstrap 95% CI \([+0.2,+4.1]\)
- Optimality: +2.2 points, bootstrap 95% CI \([+0.4,+4.4]\)
- Median TTS99 ratio: 0.82 among the 16 instances solved by both models

The chordal ordering change produced the larger and more consistent feasibility improvement:

- `chordal_fowler` versus `fowler`: +4.6 feasibility points
- `chordal_local` versus `full_order_local`: +5.3 feasibility points
- `chordal_local` versus `fowler`, applying both changes: +6.6 feasibility points and +0.8 optimality points

## Interpretation

The fourth combination is useful because it separates coefficient scaling from ordering sparsification. In this small experiment, the local coefficient alone was roughly neutral, while chordal sparsification substantially reduced model size and increased the feasible-sample rate. Combining both changes was the best overall configuration here. The apparent 2.2-point optimality gain from the local coefficient within the chordal model is encouraging, but 20 small instances are insufficient for a firm statistical or scaling claim.

Raw observations are in `quick_four_model_results.json`; paired statistics are in `quick_four_model_summary.json`.
