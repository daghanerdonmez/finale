# Test Guide for the Paper

This guide lists the experiments Claude ran (29 Sep–1 Oct 2026) to verify and benchmark the chordal-ordering Steiner QUBO. Full numbers and discussion are in `REVIEW_RESPONSE.md`. All paths are relative to `SteinerTreeProblemQUBO/research/`. Run every command from the repository root with `PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m SteinerTreeProblemQUBO.research.<module>`.

## 1. Models compared (2×2 ablation)

| Name in results | Ordering graph | Nonterminal parent term |
|---|---|---|
| `fowler` | complete (Fowler's code, `AlexFowler/steiner_to_bqm_alex.py`) | n·C(q,2) + (1−q)s |
| `full_order_local` | complete | (d⁺+1)·C(q,2) + (1−q)s |
| `chordal_fowler` | chordal completion of G − r | n·C(q,2) + (1−q)s |
| `chordal_local` | chordal completion of G − r | (d⁺+1)·C(q,2) + (1−q)s |

- All four models are built in `steiner_to_bqm_chordal.py`, except `fowler`, which comes from Fowler's code.
- `chordal` (usage-bit parents), `big_m` (the original depth model) and the comparator, domain-wall and circuit models appear only in the early pilots.

## 2. Correctness evidence (not just tests)

- **Exhaustive check.** On 7 small graphs with chordless 4- and 5-cycles, every assignment was enumerated (up to 2²¹) for the three chordal parent modes and for Fowler.
- **Zero set.** For every edge selection, the minimum penalty over the auxiliary variables is 0 exactly when the arcs form a valid rooted tree.
- **Penalty values.** Penalties are always nonnegative integers with a smallest positive value of 1, so P > U is sufficient.
- **Unit tests.** The 23 tests in `test_*.py` pass.
- **Where it lives.** The exhaustive check was an ad hoc script and is not saved in the repository. Re-implement it if the paper needs it as an artifact; the method is described in `REVIEW_RESPONSE.md` §2.

## 3. Common methodology

- **Integer costs everywhere.** This is critical, and the paper should state it.
  - With fractional costs (k/20) and P = 1.05U, floating-point residue (about 4e-16) in the Ising fields made Neal's automatic β-range degenerate (cold β ≈ 1e15).
  - It hit Fowler on 17/18 instances vs chordal on 12/18, which biased the original pilot (`pilot_results.json`, `refinement_results.json`, `larger_results.json`). **Do not use those three files.**
  - All later runs use integer weights; every run logs `beta_range`, and none was degenerate.
- **Penalty.** P = ceil(1.05·U), where U is the cost of an MST with nonterminal leaves pruned. P > U guarantees exactness, and the known optimum is never used to set it.
- **Metrics.** Each sample's selected arcs are decoded independently of the auxiliary bits.
  - *Valid/feasible:* the arcs form a rooted arborescence spanning all terminals.
  - *Optimal:* valid, and the cost equals the optimum.
  - *Best gap:* the best valid tree's cost divided by the optimum, minus 1. No valid tree at all counts as a 100% gap.
  - *TTS99:* (time per read) × ln(0.01) / ln(1 − p_opt).
- **Statistics.** Comparisons are paired per instance, with 95% bootstrap confidence intervals and Wilcoxon signed-rank p-values (`analyze_*.py`). Pooled rates alone are not the evidence.
- **Solvers.**
  - Neal SA with its automatic schedule.
  - OpenJij SA, run one read per call with distinct seeds. OpenJij 0.10.17 reuses one seed across a multi-read batch; see `diagnose_openjij_seed.py`.
  - OpenJij SQA on the Ising model normalised to max |bias| = 1 (hardware-style), with untuned defaults β = 5, Γ = 1, trotter = 4.
- **Equal budgets.** Every model gets the same reads and sweeps. Runs used 7 parallel workers, so wall-clock times are comparable within a run but are not absolute.

## 4. Experiments and files

| # | Experiment | Script | Results | Size |
|---|---|---|---|---|
| A | Integer-cost rerun of the original pilot (ladder and 3-regular graphs) | `benchmark_repairs --integer-costs` | `integer_pilot_results.json` | 18 instances, n = 8–16, 10 models |
| B | Extended ladder and 3-regular | same | `integer_extended_results.json` (analysis: `analyze_integer.py`) | 60 instances, n = 8–16 |
| C | Larger ladder and 3-regular | same | `integer_larger_results.json` | 8 instances, n = 24/32; Gurobi optima |
| D | General families, Neal and OpenJij SA | `benchmark_general` | `general_results.json` (analysis: `analyze_general general_results.json`) | 200 instances: 5 families × n = 8/12/16/20 × 25%/50% terminals × 5 seeds |
| E | Same, larger graphs | `benchmark_general --sizes 24 32 40 --sweeps 5000` | `general_large_results.json` | 150 instances |
| F | SQA on the D instances | `benchmark_general --solvers openjij_sqa` | `general_sqa_results.json` | 200 instances |
| G | SteinLib B (Beasley) | `benchmark_steinlib b01 …` | `steinlib_results.json` (b01–b12), `steinlib_large_results.json` (b13–b18); analysis: `analyze_steinlib` | 18 instances, n = 50/75/100, 100 reads × 5000 sweeps |

**Notes on D–G.**
- **Families in D–F.** They come from the project's own generators:
  - `sparse`: a spanning tree plus extra edges with p = 0.1;
  - `er30` and `er60`: Erdős–Rényi with p = 0.3 and 0.6;
  - `geo_knn4`: geometric k-nearest-neighbour with k = 4;
  - `grid`: grid graphs.
- **Optima in D–E.** Exact, by subset/MST enumeration, or by Gurobi when more than 16 vertices are optional.
- **Optima in G.** SteinLib's published optima (hard-coded in `benchmark_steinlib.py`). The project's Gurobi flow ILP did not finish in 10 minutes on b01, so it is unusable at this scale. The code raises an error if any sampled tree beats the published optimum; none did.
- **Downloaded files.** The SteinLib files are in `steinlib/B/`.

## 5. Headline results (chordal_local vs fowler)

| Setting | Valid-tree gain | Optimal / quality | Speed |
|---|---|---|---|
| B: ladder and 3-regular, 60 instances (Neal) | +21 points (better on 50 / worse on 7) | optimal +4 points (p = 2e-4) | TTS 0.47× |
| D: general n ≤ 20 (Neal / OpenJij SA) | +7 / +5 points | optimal +1.2 / +3.6 points (p ≤ 5e-4) | TTS about 0.5× |
| E: general n = 24–40 (Neal / OpenJij SA) | +24 / +20 points | almost no model finds optima; best gap −7.4 points with Neal, not significant with OpenJij SA | — |
| G: SteinLib B, 18 instances | more valid trees on 18/18 (both solvers); Fowler found no valid tree in 8/36 runs | best tree better on 15/18 (Neal), 10/18 (OpenJij SA); optimum found once in 144 runs | 10–25× faster per run; 23–49× fewer interactions |
| F: SQA, n ≤ 20 | all models under 5% valid | gain comes from the local parent coefficient, not from sparsification | — |

**Patterns that hold across experiments.**
- **The gain tracks model size.** The valid-tree gain scales with how much smaller the chordal ordering graph is (Spearman ρ = −0.83 in E).
- **Dense graphs show no gain.** The gain vanishes on dense `er60`, where chordal's ordering graph is 74–84% of the complete one.
- **Speed.** Chordal is never slower.

## 6. Claims the data does NOT support

- **Better optimisation in general.** Best-tree quality is solver- and family-dependent. Under OpenJij SA on `geo_knn4` (n ≥ 24) and SteinLib b10–b12/b16, Fowler's best trees were better despite chordal finding more valid trees.
- **Annealing as a competitive Steiner-tree method.** Beyond about 20 vertices, no QUBO model reaches optima reliably; best trees are 10–100% above optimal.
- **Anything about quantum hardware or quantum advantage.** SQA was untuned, and nothing ran on hardware.
- **Novelty of elimination-based acyclicity.** Cite Rankooh & Rintanen (AAAI 2022). The contribution is the Steiner/QUBO adaptation plus the local parent coefficient.

## 7. Known caveats to state in the paper

- **One budget per size.** One sweep budget per size and one penalty factor (1.05) were used. No sensitivity analysis was done.
- **Equal sweeps, not equal time.** Given chordal's 10–25× speed, an equal-time comparison is the most important missing experiment.
- **SQA timings.** For about the first 15 instances, SQA timings are inflated by about 10 minutes of CPU contention. Its success rates are unaffected.
- **Embedding.** No minor-embedding or qubit counts on Pegasus/Zephyr hardware layouts were computed (suggested future work).
- **Not Claude's work.** `QUICK_FOUR_MODEL_RESULTS.md`, `quick_four_model_*.json` and `make_chordal_guide_plot.py` were not produced in these runs; check their provenance separately.
