# Independent Review Response: Chordal-Ordering Steiner QUBO

Reviewer: Claude (Opus 5.5), 29 September 2026
In response to: `LLM_REVIEW_PACKET.md`

## 1. Summary

- **The math is correct.** The chordal formulation is exact. The code matches the displayed Hamiltonians.
- **The original pilot had a confounder.** Neal's automatic temperature schedule was broken by floating-point residue, and it hurt Fowler more than chordal.
- **The confounder did not explain the gap.** After fixing it, chordal still beats Fowler. On broader graph families the advantage is smaller than the pilot suggested, and it disappears on dense graphs.

## 2. Mathematical verification

- **Exhaustive check.** On 7 small graphs, I enumerated every assignment (up to 2^21), including graphs with chordless 4- and 5-cycles away from the root. I did this for chordal with each parent mode (usage, Fowler, local) and for the local Fowler baseline. For every edge selection, the minimum penalty over the auxiliary variables was 0 exactly when the selected arcs form a valid rooted tree (0 mismatches). Penalties were always nonnegative integers, and the smallest positive penalty was 1. So P > U is sufficient for every chordal mode.
- **Proofs.** Sections 8.1–8.3 are correct.
  - In §8.1, state that a shortest directed cycle is simple and has length ≥ 3, because one ordering bit per pair rules out 2-cycles.
  - Say "the orientation contains a directed triangle", not "every cycle contains one".
- **Local coefficient.** α = d⁺ + 1 is sufficient and tight for this form: α = d⁺ gives zero penalty at q = 2, s = d⁺.
- **Other claims checked.** All correct:
  - the PM² and P·a_v·4^b barriers;
  - the big-M zero set, with slack width B+1 covering [0, 2R];
  - the bounds q = O(nw) and τ = O(nw²);
  - the P > c_max counterexample;
  - the subset/MST reference and the Gurobi flow ILP.
- **Gurobi caveat.** The Gurobi reference's "OPTIMAL" status with the default MIP gap is exact only because the costs lie on a coarse grid. Set `MIPGap=0` before calling it certified.
- **Numbers.** Every number quoted in the packet matches the saved JSON files. The 32-vertex ladder model sizes (148, 555 and 812 variables) also check out by hand.
- **Minor.** §6 drops the word "integer-valued" when discussing separately weighted components.

## 3. Confounder in the original benchmark

- **Cause.** With costs k/20 and P = 1.05U, converting the BQM to Ising form leaves field residues of about 4e-16 where the value should be exactly 0.
- **Effect.** `neal.default_beta_range` treats these residues as real energy gaps and sets the cold end to β ≈ 1e15–1e16 (a sane value is about 8–50). The schedule is geometric, so most sweeps run fully frozen.
- **Uneven impact.** In the 18-instance pilot:

| Model | Instances with degenerate schedule | Share of sweeps below β = 50 |
|---|---:|---:|
| Fowler | 17/18 | about 25% |
| Chordal | 12/18 | about 46% |
| Big-M | 18/18 | — |

- **Fix.** Scale costs by 20 so they are integers and use P = ceil(1.05U). All coefficients are then exact, and the coldest β is at most 1.5 for every model. The flag `--integer-costs` was added to `benchmark_repairs.py`, and each run now logs `beta_range`.

## 4. Reruns with the fix (Neal SA)

Same instances, integer costs. Share of samples:

| Set | Fowler: valid tree / optimal | Chordal-local: valid tree / optimal |
|---|---|---|
| Original pilot, 18 instances | 48% / 32% | 72% / 37% |
| Extended, 60 instances (ladder and 3-regular, n = 8–16) | 57% / 29% | 77% / 34% |

- **Paired statistics on the 60 instances.** Chordal-local vs Fowler:
  - valid trees: +21 points (better on 50 instances, worse on 7, p ≈ 1e-9);
  - optimal trees: +4 points (better on 41, worse on 13, p = 2e-4);
  - median TTS99 (wall clock) 0.47× Fowler's.
- **n = 24/32.** Fowler found no valid tree on any ladder instance. Chordal found trees 0–10% above optimal. Every model hit the exact optimum in under 1% of samples.

Files: `integer_*_results.json`, `analyze_integer.py`.

## 5. General tests

- **Instances.** 200, from the project's own generators with integer weights 1–100:
  - families: sparse (tree plus 10% extra edges), Erdős–Rényi p = 0.3 and p = 0.6, geometric kNN (k = 4), and grid;
  - n = 8, 12, 16, 20; terminals 25% or 50%; 5 seeds each;
  - exact optima from subset/MST enumeration.
- **Models.** A 2×2 ablation: {full, chordal} ordering × {Fowler, local} parents.
- **Solvers.** Neal SA and OpenJij SA (per-read seeds), 100 reads × 2000 sweeps. No degenerate schedules occurred.

Chordal-local minus Fowler, paired by instance:

| | Neal SA | OpenJij SA |
|---|---|---|
| Valid tree | +7.3 points [5.7, 9.0], better on 128 / worse on 32 | +5.3 points [3.9, 6.8], better on 112 / worse on 44 |
| Optimal tree | +1.2 points [0.4, 2.0], better on 94 / worse on 43, p = 5e-4 | +3.6 points [2.4, 4.9], better on 103 / worse on 45, p = 6e-8 |
| Median TTS99 ratio | 0.51 | 0.53 |

Square brackets are 95% bootstrap intervals. Both ablation directions contribute:
- **Chordal/Fowler vs Fowler** (sparsification alone): valid +6.8, optimal +1.0 (Neal).
- **Chordal/local vs full/local**: valid +6.2, optimal +0.7 (Neal, p = 0.13 — not significant).

By family (Neal). "Size" is chordal ordering edges divided by complete ordering edges:

| Family | Size | Valid-tree gain | Optimal-tree gain |
|---|---:|---:|---:|
| Grid | 0.34 | +14.6 | +0.6 |
| Sparse | 0.30 | +8.3 | +4.0 (p = 0.002) |
| Geometric kNN | 0.49 | +8.9 | +0.5 |
| ER p = 0.3 | 0.38 | +4.4 | +1.6 |
| ER p = 0.6 | 0.74 | +0.2 | −0.7 (no difference) |

- **Advantage tracks model size.** Spearman correlation between the size ratio and the valid-tree gain: ρ = −0.45 (Neal) and −0.33 (OpenJij SA).
- **Advantage grows with n.** The valid-tree gain is +16 points at n = 20 (Neal).
- **Harder instances.** 129 instances are ones where the pruned MST is not already optimal. On these the valid-tree gain is +8.6 points (Neal); the optimal-tree gain is +2.5 points with OpenJij SA (p = 4e-4) and +0.3 with Neal (not significant).
- **Optimum finding is weak for everyone at this scale.** At n = 20, every model hits the exact optimum in about 2% of samples, and the best trees are 7–13% above optimal.

Files: `benchmark_general.py`, `analyze_general.py`, `general_results.json`.

## 6. Larger synthetic graphs (n = 24, 32, 40)

150 instances: same five families, 25%/50% terminals, 5 seeds each, 100 reads × 5000 sweeps. Optima come from enumeration or Gurobi.

| Chordal-local minus Fowler | Neal SA | OpenJij SA |
|---|---|---|
| Valid tree | +23.8 points, better on 122 / worse on 13 | +19.6 points, better on 112 / worse on 25 |
| Best-tree gap to optimum | 7.4 points lower (better on 95 / worse on 46, p = 2e-6) | 1.4 points lower (not significant) |

- **The feasibility gain tracks model size.** Spearman ρ = −0.83 between the size ratio and the gain. Grid and geometric kNN (size about 0.20): +39 to +52 points. ER p = 0.6 (size 0.84): about 0.
- **Nobody finds optima at this size.** Every model hits the exact optimum in under 0.5% of samples, and mean best-tree gaps run from 9% (grid, chordal) to about 100% (ER p = 0.6).
- **More valid trees ≠ better trees.** Under OpenJij SA on geometric kNN, chordal finds valid trees +45 points more often, yet Fowler's best tree is better on 27/30 instances (mean gap 16% vs 26%). The same happens on ER p = 0.6. Chordal's best-tree quality advantage holds on grids under both solvers (about 9% vs 21%), but not universally.

Files: `general_large_results.json`.

## 7. SteinLib test set B (Beasley; 18 instances, n = 50–100)

- **Reference optima.** I used SteinLib's published (proven) optima. The project's Gurobi flow ILP made no progress in 10 minutes on b01, so it is not usable as a reference at this scale.
- **Budget.** 100 reads × 5000 sweeps. A sampled tree cheaper than the published optimum would raise an error; none did.
- **Model sizes.** The chordal ordering graph is 4–17% of the complete one (elimination width 4–22).

| n | Fowler: variables / interactions / seconds per run | Chordal-local: variables / interactions / seconds per run |
|---|---|---|
| 50 | 1,373 / 56,377 / 32 s | 396 / 2,413 / 3 s |
| 75 | 2,997 / 196,014 / 97 s | 672 / 5,364 / 6 s |
| 100 | 5,244 / 472,577 / 318 s | 965 / 9,578 / 13 s |

- **Valid trees.** Chordal-local found more valid trees than Fowler on 18/18 instances under both solvers.
  - Fowler found no valid tree at all in 8 of 36 instance–solver runs, including b14–b16 under both solvers. Chordal always found one.
  - Example, b07: Fowler 0–1% valid samples vs chordal 7–15%.
- **Best tree.** Chordal-local was strictly better on 15/18 instances with Neal and 10/18 with OpenJij SA. With OpenJij SA on the denser b10–b12 and b16, chordal's best trees were worse (for example, b11: 89% gap vs 42%).
- **Optima.** The exact optimum was found once in 144 runs: b01, chordal-local, Neal. Typical best-tree gaps are 5–60%.
- **Conclusion.** On real benchmark instances, sampling-based QUBO solving is far from competitive with either formulation at n ≥ 50. Chordal is 10–25× faster per run and much more reliable at producing a valid tree.

Files: `benchmark_steinlib.py`, `analyze_steinlib.py`, `steinlib_results.json`, `steinlib_large_results.json`, `steinlib/B/`.

## 8. Simulated quantum annealing (OpenJij SQA)

Same 200 instances as §5 (n = 8–20). The Ising model was normalised to max |bias| = 1, as hardware autoscaling does. Untuned defaults: β = 5, Γ = 1, trotter = 4; 100 reads × 2000 sweeps.

- **Every model essentially fails.** Mean valid-tree rates are 1.5–4.3%, optimal rates 0.1–0.8%, and about 0 at n ≥ 16.
- **Chordal-local vs Fowler.** Valid trees +2.8 points (better on 75 / worse on 9), optimal +0.7 points (better on 27 / worse on 0, p = 2e-8).
- **This comes from the parent coefficient, not the sparsification.**
  - Chordal/Fowler vs Fowler: +0.0 points.
  - Full/local vs Fowler: about the same gain as chordal/local.
  - Normalisation divides everything by Fowler's n·P coefficient, which crushes the cost and penalty gaps. The local coefficient (≤ deg + 1) avoids that.
  - This is the one setting where the coefficient-range argument in packet §5 shows up clearly.
- **Caveat.** The SQA parameters were not tuned, so this is a weak test of SQA itself. Timings for about the first 15 instances are inflated by about 10 minutes of CPU contention; success rates are unaffected.

Files: `general_sqa_results.json`.

## 9. Not done

- **Quantum hardware.** No hardware runs.
- **SQA tuning.** No SQA parameter tuning.
- **Novelty.** No literature search beyond the packet's own; the Rankooh–Rintanen framing still applies.
- **Classical baselines.** Not compared, because they are not needed for the formulation claim. Gurobi on a good ILP, or SCIP-Jack, would beat every QUBO here.

## 10. Recommended claim

> The chordal-completion sparsification of Fowler's ordering QUBO is exact. Its size is controlled by elimination width, and it reduces to Fowler in the worst case. On SteinLib B it is 23–49× smaller (in interactions) and 10–25× faster per annealing run. Across 350 synthetic instances (n = 8–40) and all 18 SteinLib B instances, simulated annealing finds valid Steiner trees significantly more often with it. The gain grows with sparsity and size, and vanishes on dense graphs. Improvements in best-solution quality are smaller and solver-dependent. For hardware-like normalised annealing, the decisive change is the local parent coefficient (avoiding Fowler's n-scaled term), not the sparsification. No QUBO formulation tested reaches optimal solutions reliably beyond about 20 vertices.

**Corrections to the packet.**
- Replace the §14 pilot numbers with the integer-cost reruns.
- Add the schedule confounder to §16.
- Present the chordal advantage as primarily a feasibility, size and speed advantage, not an optimality advantage.
- Present the local parent coefficient as a separate contribution that matters under coefficient normalisation.
