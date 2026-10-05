# A feasible direction for the Steiner-tree QUBO project

Research note, 23 September 2026. Based on the current `491finalreport.tex`, the implementation, explicit derivations, and new reproducible experiments. The report and existing formulation files have not been edited.

## Assessment

**The depth-based feasibility idea is correct. Its binary big-M implementation is a poor match for local annealing moves, for a reason stronger than simply “large coefficients.”** Coefficient tuning alone cannot remove its large barrier-to-violation-gap ratio while preserving the same general exactness guarantee.

There are two defensible directions:

1. **Repair the compact rank formulation with bitwise comparison circuits.** This removes big-M and binary place values from the penalties. It gives an exact formulation with logarithmic rank storage and small quadratic coefficients. I implemented and tested it. It helps on small instances, but does not resolve the larger-instance annealing problem in this pilot.
2. **Make ordering sparse using a chordal completion.** Keep the ordering model's useful energy structure, but encode only the comparisons needed by the graph and its completion. This has a clear correctness proof, much smaller models on low-width graphs, and the strongest empirical results here. This is my recommended main project direction, with the compact comparator model as an informative comparison.

Neither direction establishes quantum advantage or universal annealing superiority. The general techniques have prior art. A defensible contribution would be their exact Steiner-specific construction, coefficient and interaction analysis, and controlled empirical comparison.

## 1. What is actually wrong with the original encoding?

Write the residual for an arc as

\[
r_{uv}=o_v-o_u-1+M(1-e_{uv})-g_{uv}.
\]

At a penalty-free assignment, every residual is zero. If only the edge bit changes, the residual changes by exactly ±M. Therefore

\[
\boxed{\Delta H_{\mathrm{depth}}=\lambda M^2.}
\]

This is an actual energy barrier from a valid encoding, not merely an inference from the largest matrix entry. Changing an edge while maintaining feasibility ultimately requires the slack and possibly depths to change with it. A single-bit sampler initially pays this large penalty. From a fully penalty-free state the other nonnegative constraints cannot offset it; removing the edge saves only its cost.

If bit b of a nonroot rank changes while all witnesses remain fixed, every incident arc contributes λ4^b. If a_v is the number of allowed directed arcs incident to v,

\[
\Delta H_{\mathrm{depth}}=\lambda a_v4^b.
\]

Inactive arcs contribute too. They impose no logical ordering restriction after their slack is optimized, but their *current* slack still resists changes to either endpoint. Binary carries compound this: changing a rank from 7 to 8 requires four bit flips, and every affected certificate must adjust.

These statements explain why having fewer interactions need not help. The model is compactly describing a global relation using certificates that are expensive to update locally.

**What scaling can and cannot do.** Dividing the entire Hamiltonian by a constant leaves its relative barriers unchanged; for ideal classical Metropolis dynamics, compensating the inverse-temperature schedule gives identical transition probabilities. Scaling can nevertheless help when a solver's fixed schedule was badly matched to the absolute coefficient scale. Dividing only H_depth by M² reduces its smallest positive violation penalty by the same factor. Restoring the original worst-case feasibility guarantee restores the scale problem. Unary slack alone also leaves the M-sized edge switch intact.

This is not an impossibility theorem for the original Hamiltonian under every algorithm. Coordinated moves, repair methods, or different schedules may help. It is a concrete structural objection to expecting ordinary single-bit annealing to be rescued by global normalization.

## 2. Correctness and safe penalty selection

Your parent/usage constraints, together with strict rank increase on every selected arc, have the right zero set. A selected directed cycle is impossible. Every terminal except the root has one parent. A nonterminal with children has a parent, and no vertex has multiple parents. Following the parent chain therefore reaches the only possible root. Conversely, orient any feasible tree away from the root and use its graph distances as ranks; all penalties can be zero.

For nonnegative costs and nonnegative integer-valued unweighted penalties, a simple sufficient choice is

\[
\boxed{\lambda>U,}
\]

where U is the cost of any known feasible tree. Every infeasible assignment then costs at least λ>U, and some feasible assignment costs U. A pruned spanning tree provides U without knowing the optimum. This bound is conservative; empirically smaller weights may work, but require separate certification or an explicitly heuristic interpretation. With distinct penalty weights, making each exceed U is a sufficient common argument, not necessarily the sharpest one.

There is no universal replacement λ>c_max for these parent constraints. On a path with only the two endpoints terminal and all edge costs c, the optimum is (n−1)c. The empty edge set can satisfy all the depth/slack/usage constraints and violate only the nonroot terminal's parent constraint, costing λ. Thus λ<(n−1)c can prefer an infeasible solution even though λ>c.

For the implementation's B=ceil(log₂n), R=2^B−1, M=R+1, inactive slack can reach 2R and needs B+1 bits. With A=2m−d_r, the current builder actually has

\[
A(B+2)+nB+(n-k)
\]

variables. The report's formula A(B+1)+nB+(n−k) misses one slack bit per arc for this implementation. Its asymptotic result is unchanged. Fixing root bits to zero rather than penalizing them is a valid minor simplification.

Similarly, with B+1 slack bits a depth square contains 3B+2 variables, not 3B+1. Adding the clique sizes of all squares counts contributions before aggregation, rather than distinct final BQM interactions: depth-bit pairs recur in several arc penalties, and root-depth pairs overlap too. The displayed exact interaction-count equality should therefore be corrected or labeled as a contribution bound. The O(m log²n + Σdeg²) upper bound still holds; the new experiment counts use actual nonzero assembled interactions.

## 3. Exact compact repair: bitwise comparison

For each selected-arc comparison, compute the unsigned subtraction o_u−o_v one bit at a time. Let b_i be the borrow into position i, d_i the difference bit, and b_0=0. Use

\[
C_{uv}=\sum_{i=0}^{B-1}
(o_{u,i}-o_{v,i}-b_i-d_i+2b_{i+1})^2.
\]

Each equation involves coefficients only ±1 and 2. At zero penalty it is exactly a binary subtractor, and the final borrow satisfies

\[
b_B=1\iff o_u<o_v.
\]

Replace the big-M depth term by

\[
\lambda\sum_{(u,v)\in A}\big[C_{uv}+e_{uv}(1-b_B)\big].
\]

Every term is nonnegative and integer-valued. A selected edge with incorrect order cannot have zero penalty. Conversely, valid ranks have consistent subtraction witnesses. No extra hierarchy between gate penalties is required for the sufficient λ>U proof.

The assembled quadratic coefficients in this prototype have magnitude at most 4λ. Shared ranks still accumulate linear coefficients and interaction degree proportional to graph degree; “small gate coefficients” must not be described as constant degree or constant total local field on arbitrary graphs.

With the report's parent squares, counts are O((n+m)log n) variables and O(m log n + Σ_v deg(v)²) interactions. Optional sequential parent counters remove the parent cliques: impose binary prefix equations s_i−s_{i−1}−e_i=0, with the final prefix fixed to 1 for a terminal or p_v for a nonterminal. This gives O((n+m)log n) variables **and** interactions for the complete circuit model. The experiments use ordinary parent squares to isolate the change in ordering.

A useful exact contrast: flipping a rank bit with fixed witnesses now costs λa_v, independent of the bit position, instead of λa_v4^b. Flipping an edge no longer incurs an M² term. But changing ranks still requires updating multiple circuit witnesses, even for unselected arcs. This residual issue is visible in the experiments.

### Sharing comparisons and avoiding root circuits

The refined prototype uses one comparator per nonroot undirected edge, rather than one per allowed arc. Let i(v) be a fixed distinct vertex index and order nonroot vertices lexicographically by (rank(v),i(v)). For endpoints u,v with i(u)<i(v), compute c=[rank(v)<rank(u)] and add

\[
e_{uv}c+e_{vu}(1-c).
\]

Equal numerical ranks are safe because the index breaks ties. This is an ordering formulation with binary ranks, rather than literal strict numerical depth along every edge. A selected arc respects a genuine total order, so directed cycles remain impossible. Any tree can be represented, for example by distinct topological ranks. Root edges need no comparison because there are no incoming root arcs.

For B'=ceil(log₂(n−1)), m₀=m−d_r, and square parents, the variable count is

\[
A+(n-k)+(n-1)B'+2m_0B'.
\]

This substantially improves the constants and remains exact. It still performed poorly on the larger pilot, despite its much smaller coefficient range. That is direct evidence that coefficient range is not the whole annealing problem.

Files: `steiner_to_bqm_circuit.py`, `steiner_to_bqm_binary_order.py`, and their tests.

## 4. Recommended direction: sparse chordal ordering

Let G₀ be the graph induced by the nonroot vertices. Construct an undirected **chordal completion** H of G₀ by eliminating vertices and connecting their remaining neighbors into cliques. The implementation uses a deterministic greedy minimum-fill heuristic. Fill edges are ordering scaffolding; they are never eligible Steiner-tree edges and contribute no edge cost.

Introduce x_uv only for edges of H. Retain the usual edge-order consistency penalties on actual graph edges. For every triangle u<v<w in H add the same quadratic polynomial used by Fowler:

\[
T_{uvw}=x_{uw}+x_{uv}x_{vw}-x_{uv}x_{uw}-x_{uw}x_{vw}.
\]

This takes value 1 for a cyclic triangle orientation and 0 otherwise.

**Why it is sufficient.** Suppose an orientation of H has a directed cycle, and choose a shortest one. If its length is at least four, chordality supplies a chord. Whichever way that chord points, it closes a shorter directed cycle using one of the two directed paths around the original cycle. Contradiction. Thus every cyclic orientation contains a directed triangle. Eliminating directed triangles therefore eliminates every directed cycle.

**Why it does not exclude a Steiner tree.** Orient the tree away from the root, extend the selected arcs to any total order of nonroot vertices, and orient every completion edge according to that order. All triangle and edge-order penalties vanish. Parent constraints finish the connectivity argument.

Simply dropping nonedge comparisons *without* completing the graph is incorrect: a chordless directed four-cycle has no triangle to penalize.

Let q=|E(H)|, τ be the number of triangles, and w the maximum number of later neighbors in the **chosen** elimination order. Then

\[
q\le(n-1)w,\qquad
\tau\le(n-1)\binom{w}{2}.
\]

The p-variable version uses A+q+(n−k) variables. The variants without p use A+q. Triangle penalties contribute exactly 3τ distinct quadratic interactions; total interactions are O(nw² + m + Σ_v deg(v)²).

Thus bounded degree and bounded elimination width give linear-size models. This is a stronger improvement on an appropriate graph class than merely reducing n³ to n log²n. **It is not a guarantee for every sparse graph.** Sparse graphs can have large treewidth and large fill. The measured width of a greedy completion is an upper bound, not a claim that minimum treewidth was computed. Wider graphs, grids, and dense instances need explicit evaluation.

### Parent constraints: retain an informative ablation

The prototype has three parent options:

- `usage`: the report's p-variable squares and no-fake-root terms.
- `fowler`: Fowler's original grouped nonterminal penalty with coefficient n, so sparsification is the only mathematical change from the baseline.
- `local`: eliminate p and replace the global n coefficient by a valid vertex-local bound.

For the last option, let q_v be selected indegree, s_v selected outdegree, and d_v⁺ the number of *allowed* outgoing arcs. Set α_v=d_v⁺+1 and use

\[
F_v=\alpha_v\binom{q_v}{2}+(1-q_v)s_v.
\]

At q_v=0 this equals s_v, at q_v=1 it is zero, and at q_v≥2,

\[
F_v=(q_v-1)(\alpha_v q_v/2-s_v)\ge1,
\]

because s_v≤d_v⁺<α_v. Therefore the grouped penalty is nonnegative and has exactly the desired zero set. This reduces the nonterminal multiple-parent coefficient from n to d_v⁺+1. The fake-root factor alone can be negative; these two parts must be kept together when proving bounds or adjusting weights.

On degree-three graphs the local coefficient is at most 4, independent of n. The local variant has more edge-edge couplings than the p formulation, but fewer variables, and gave better optimal-sample rates in this pilot. Removing p alone is not enough: the full-order/local-parent ablation did not improve on the full Fowler baseline overall.

**An important coefficient subtlety.** Under x=(1+s)/2, the triangle penalty becomes

\[
T=(1+s_xs_y-s_xs_z-s_zs_y)/4.
\]

It has no Ising linear fields. Large accumulated QUBO linear biases from ordering triangles cancel when converted to Ising. Compare actual Ising fields/couplers and local move energies as well as binary QUBO coefficient ranges; raw maximum linear bias alone can be misleading.

File: `steiner_to_bqm_chordal.py`.

## 5. Domain-wall alternative: valid, but not the leading candidate

For D=n−1, encode z_{v,j}=[o_v≥j] and penalize nonmonotone thresholds by z_{v,j+1}(1−z_{v,j}). A selected arc must satisfy e(1−z_{v,1})=0, e z_{u,D}=0, and e z_{u,j}(1−z_{v,j+1})=0 at internal levels. With a new bit a, the internal cubic has the exact quadratization

\[
ez(1-w)=\min_{a\in\{0,1\}}\{ez+a(2-e-z-w)\}.
\]

The quadratic expression is nonnegative for every assignment. This gives an exact QUBO with maximum quadratic coefficient 2λ and O(nD+mD+Σdeg²) interactions. It keeps an O(n²) interaction advantage over full ordering on bounded-degree sparse graphs, but sacrifices logarithmic depth storage.

Its poor pilot results show why “use unary/domain-wall depths” is not sufficient advice by itself. Each edge is linked to many level gadgets; auxiliary coordination and interaction degree still matter. The implementation is a controlled negative result, not my recommended main model.

File: `steiner_to_bqm_domain_wall.py`.

## 6. Experiments actually run

The benchmark stores every graph, terminal set, exact optimum, penalty, count, coefficient range, and aggregated sampling result in JSON. It does not use the known optimum to set penalties. Costs are positive multiples of 1/20. U comes from an MST pruned of nonterminal leaves, and every model uses the safe common λ=1.05U. Initial states are random; there is no feasibility repair or postprocessing into a tree.

The reference optimum is certified by exhaustive enumeration of optional vertex subsets with an MST on each connected induced subgraph, or by the existing exact flow ILP for larger instances. The subset/MST method is exact: every Steiner tree uses some enumerated vertex set and has cost at least that set's MST, and each connected set's MST is a feasible tree.

Each returned sample is classified using the selected directed arcs. “Feasible” means a rooted arborescence spanning every terminal, with no disconnected selected component. “Optimal” additionally means actual selected-edge cost equals the exact optimum. “Zero penalty” separately checks full BQM energy minus directed edge cost. Samples with bad auxiliary bits can therefore count as decoded feasible but not as certified QUBO solutions. Each model receives its full fixed read budget; no stopping on first hit.

### Small pilot

18 instances: ladders and connected random 3-regular graphs; n=8,12,16; three seeds for each size/family. 100 reads × 2,000 sweeps per model with `neal.SimulatedAnnealingSampler`, its automatic temperature schedule, and recorded deterministic seeds. Ten variants were examined; the refinement variants were chosen after the initial results. This is exploratory evidence, not an untouched test set or a significance claim.

| Model | Instances with optimum | Feasible samples | Optimal samples | Zero-penalty samples |
|---|---:|---:|---:|---:|
| Original big-M depth, corrected common weight | 0/18 | 0.3% | 0.0% | 0.1% |
| Full ordering / Fowler parents | 15/18 | 41.4% | 33.3% | 41.4% |
| Full ordering / p parents | 12/18 | 41.5% | 19.3% | 41.5% |
| Bitwise strict-depth circuit | 10/18 | 13.9% | 9.9% | 3.5% |
| Shared binary-order comparator | 12/18 | 25.1% | 13.9% | 16.2% |
| Domain-wall depth | 4/18 | 3.9% | 3.8% | 3.9% |
| Chordal ordering / p parents | 15/18 | 67.2% | 24.3% | 67.2% |
| Chordal ordering / Fowler parents | 17/18 | 65.8% | 38.3% | 65.8% |
| Chordal ordering / local parents | 16/18 | 66.8% | 39.0% | 66.8% |
| Full ordering / local parents | 15/18 | 39.4% | 30.5% | 39.4% |

The parent ablation matters: attributing the difference between full Fowler and chordal/p entirely to sparse ordering would be misleading. Chordal/Fowler provides the cleaner direct comparison. Local weighting further improves coefficient scaling; it does not uniformly dominate on every performance metric.

### Larger pilot, including a regime where big-M already has fewer interactions

Eight further instances: the same two families, n=24,32, two seeds each; 100 reads × 5,000 sweeps, same safe-weight policy and sampler. There are too few instances to infer asymptotic solver behavior.

| Model | Instances with optimum | Feasible samples | Optimal samples |
|---|---:|---:|---:|
| Big-M depth | 0/8 | 0.0% | 0.0% |
| Full ordering / Fowler | 0/8 | 10.2% | 0.0% |
| Shared comparator | 0/8 | 0.6% | 0.0% |
| Chordal / p | 0/8 | 66.0% | 0.0% |
| Chordal / Fowler | 2/8 | 52.1% | 2.1% |
| Chordal / local | 3/8 | 50.4% | 2.8% |

Finding optimal trees is still difficult. The strong conclusion is improved feasibility and model size on these graphs; an overall optimization breakthrough is not established.

For the particular 32-vertex ladder with seed 0:

| Model | Variables | Interactions | Largest quadratic magnitude / λ |
|---|---:|---:|---:|
| Big-M | 812 | 9,874 | 2,048 |
| Full Fowler | 555 | 13,825 | 32 |
| Shared comparator | 707 | 2,326 | 4 |
| Chordal / p | 170 | 386 | 2 |
| Chordal / Fowler | 148 | 424 | 32 |
| Chordal / local | 148 | 424 | 4 |

Thus the original depth model already beats full ordering on interaction count in this example, yet fails empirically. Chordal/local reduces full ordering's interactions by about 32.6× and reduces big-M's largest quadratic coefficient by 512× at the same λ. The low-coefficient comparator also fails here, showing that precision repair alone is not sufficient.

![Pilot comparison](/Users/daghanerdonmez/Desktop/evvifing/boun/cmpe/491-492/finale/SteinerTreeProblemQUBO/research/comparison.png)

The figure's size/scaling panels use ladders up to 64 vertices and measure model construction, not successful annealing at those sizes. The timing values in JSON are wall-clock observations including sampler overhead; runs were not isolated for a publishable timing comparison. No SQA or hardware superiority is claimed from these SA results.

### OpenJij SA cross-check and a seed trap

I also ran four of the small instances (both families, n=8,12, seed 0) using OpenJij SA, 100 reads × 2,000 sweeps, with a **different deterministic seed for each read**. Full ordering and chordal/local both found the optimum on all four. Full ordering had 81.75% feasible and 74.5% optimal samples; chordal/local had 85.25% feasible and 74.0% optimal. The shared comparator had 49.25% feasible and 22.25% optimal; big-M had neither feasible nor optimal samples. This small check supports comparable ordering performance, not a universal win for chordal ordering. Results are in `openjij_sa_results.json`.

The distinct-seed qualification is critical. In the installed OpenJij 0.10.17, a call with `num_reads=N, seed=0` reuses the same seed for initialization and the Monte Carlo routine on each read. The newer project notebook does exactly this. A controlled diagnostic on an 8-vertex instance produced **one distinct state from eight fixed-seed reads, versus eight distinct states from eight separately seeded reads**, for both SA and SQA. Thus a seeded batch can repeat a single trajectory. This issue is independent of the QUBO formulation.

The first OpenJij cross-check exposed this problem; those repeated-trajectory data were retained separately as `openjij_sa_fixed_seed_diagnostic.json` and are not used as independent sampling evidence. `diagnose_openjij_seed.py` and `openjij_seed_diagnostic.json` reproduce the focused SA/SQA check. The benchmark now uses one call per OpenJij read with distinct recorded seeds. The Neal pilots use a different sampler and are unaffected by this OpenJij-specific behavior.

## 7. Existing implementation and reporting issues to fix

The detailed audit is in `audit_findings.md`. The most consequential points are:

- The historical benchmark calls total penalized energy “cost,” and checks optimum hits without requiring graph feasibility. Approximation ratios should use actual edge cost of valid decoded trees. Keep QUBO certificate feasibility separate.
- `SampleSet.data()` is energy-sorted by default. Its enumeration index is not a chronological first-hit read. Batch completion time also is not the true time of the first successful read.
- Stopping subsequent batches on a hit makes feasibility rates depend on different, outcome-correlated budgets. Use fixed-budget primary comparisons.
- Some old wrappers import `MyFormulization` instead of the current `MyFormulation`, and call outdated function signatures. The present checkout cannot reproduce those scripts unchanged.
- The current module's default weights are all 1, but the newer comparison notebook explicitly overrides them and normalizes costs. Do not mistake defaults for that notebook's experimental settings: its parent/usage/root weights are 5, its depth weight is 1, and Fowler's weight is 5.
- That notebook and historical benchmarks use OpenJij **SQA**, rather than ordinary SA. The pinned SQA implementation has beta=5 and gamma=1 defaults and does not automatically adapt the quartic schedule to Hamiltonian magnitude. This is an additional scale mismatch to test, not an explanation that removes the intrinsic M² barrier. Tune/log beta, gamma and normalization on a separate training set.
- The newer notebook's fixed seed is reused within a multi-read batch by this installed OpenJij version. Verify sample diversity, and use separate per-read seeds or an unseeded batch. The older benchmark wrappers do not necessarily set a seed; do not attribute this specific bug to every historical result.

These observations do not prove the old plots are wrong in every detail, and they do not dismiss your observed failures. They establish why corrected experiments are necessary before drawing quantitative comparisons from those plots.

## 8. Concrete project plan and stopping criteria

**Main question:** Can graph-aware sparse ordering preserve the useful annealing behavior of full ordering while lowering its variables, couplings, and coefficient scale?

1. Present the big-M single-flip barrier as a precise negative result and retain the existing formulation as a baseline. This turns the current difficulty into an explanatory result rather than a failed implementation.
2. Formalize chordal-ordering correctness, q/τ/width bounds, and the local-parent penalty. Compare unchanged Fowler parents first, then local parents and p parents as separate ablations.
3. Keep the shared binary comparator as the compact asymptotic alternative. It establishes that logarithmic-size ranks do not inherently require large coefficients, while its annealing behavior separates precision from witness-coordination effects.
4. Use unseen seeds and families: ladders, grids, geometric graphs, sparse random graphs with increasing measured width, and dense graphs. Choose root and elimination heuristic using graph statistics, not the test optimum. Report fill, triangle count, width, BQM degree, Ising scales, feasibility, certificate feasibility, and feasible-solution gaps.
5. Tune each sampler fairly on separate instances. Use both matched sweeps/reads and matched wall time, with repeated independent sampler seeds. Include confidence intervals based on independent instances, not a claim that correlated reads across mixed instances are identical Bernoulli trials. Compare SQA separately from SA.
6. Use ordinary STP preprocessing consistently across every formulation. Avoid restricting maximum depth without a justified bound; that can silently change the problem. If block moves or repair are introduced, label them as solver changes and apply comparable treatment to baselines.

**Decision rule:** if chordal ordering's improvement disappears on held-out families or fill becomes too large, narrow the contribution to low-width graph classes and a measured precision/landscape tradeoff. Do not keep claiming a generic sparse-graph advantage for the chordal model. If the circuit model continues to have low certificate-feasibility despite bounded coefficients, report that directly and stop treating coefficient normalization as the remaining missing fix.

Low-width STP also admits strong classical algorithms. A compact QUBO on those graphs is a formulation contribution, not evidence that annealing beats classical methods.

## 9. Reproduction and files

Run from the repository root using the existing environment:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s SteinerTreeProblemQUBO/research -p 'test_*.py'

PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m SteinerTreeProblemQUBO.research.benchmark_repairs --reads 100 --sweeps 2000 --seeds 3 --sizes 8 12 16

PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m SteinerTreeProblemQUBO.research.benchmark_repairs --reads 100 --sweeps 2000 --seeds 3 --sizes 8 12 16 --models binary_order chordal_fowler chordal_local full_order_local --output SteinerTreeProblemQUBO/research/refinement_results.json

PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m SteinerTreeProblemQUBO.research.benchmark_repairs --reads 100 --sweeps 5000 --seeds 2 --sizes 24 32 --models big_m fowler binary_order chordal chordal_fowler chordal_local --output SteinerTreeProblemQUBO/research/larger_results.json

PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m SteinerTreeProblemQUBO.research.summarize_repairs

PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m SteinerTreeProblemQUBO.research.benchmark_repairs --reads 100 --sweeps 2000 --seeds 1 --sizes 8 12 --models big_m fowler binary_order chordal_local --sampler openjij_sa --output SteinerTreeProblemQUBO/research/openjij_sa_results.json

PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m SteinerTreeProblemQUBO.research.diagnose_openjij_seed
```

The larger-reference branch requires a working Gurobi license; it was available for this run. The small pilot and the encoding tests do not require Gurobi. All completed pilot settings and graphs are also saved in the result JSON files, so solver reruns need not rely solely on regeneration.

The tests cover local truth tables, binary borrow comparisons, exact tiny-instance global minima, invalid cycles, exhaustive chordal orientations, extension of valid trees, coefficient bounds, equality of full/Fowler to the existing implementation, and the local-parent zero set. They supplement the mathematical arguments; they do not prove annealing performance.

## 10. Primary sources and novelty boundaries

- [Fowler, *Improved QUBO Formulations for D-Wave Quantum Computing* (2017)](https://www.researchgate.net/publication/322634540_Improved_QUBO_Formulations_for_D-Wave_Quantum_Computing): direct ordering baseline, author-uploaded thesis. The local implementation is the baseline actually used in the new experiments.
- [Rankooh and Rintanen, *Propositional Encodings of Acyclicity and Reachability by Using Vertex Elimination*, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20530): directly relevant elimination-based acyclicity prior art, in SAT. The construction here is an adaptation to quadratic triangle penalties and Steiner constraints; do not claim to have invented sparse elimination-based acyclicity. The targeted search did not establish whether this exact Steiner QUBO already exists.
- [Chancellor, *Domain wall encoding of discrete variables for quantum annealing and QAOA* (2019)](https://arxiv.org/abs/1903.05068), and [Chen, Stollenwerk and Chancellor, *Performance of Domain-Wall Encoding for Quantum Annealing* (2021)](https://arxiv.org/abs/2102.12224): encoding and empirical background, not evidence for the performance of this particular Steiner construction.
- [Karimi and Ronagh, *Practical Integer-to-Binary Mapping for Quantum Annealers*](https://arxiv.org/abs/1706.01945): bounded-coefficient integer encoding. Merely changing the integer expansion does not remove the original big-M edge switch.
- [Zaman, Tanahashi and Tanaka, *PyQUBO*](https://arxiv.org/abs/2103.01708): logic/arithmetic-to-QUBO background. Binary subtraction gates themselves are standard; the graph formulation and measured tradeoffs are the relevant contribution.
- [Liu and Dinneen, *Solving the Bounded-Depth Steiner Tree Problem using an Adiabatic Quantum Computer* (2019)](https://doi.org/10.1109/CSDE48274.2019.9162395): prior depth-based Steiner work to consider before broad novelty claims. Only bibliographic/abstract-level content was inspected in this investigation.
- [OpenJij 0.10.17 SQA source](https://raw.githubusercontent.com/Jij-Inc/OpenJij/v0.10.17/openjij/sampler/sqa_sampler.py), and [dimod SampleSet.data documentation](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/generated/dimod.SampleSet.data.html): source support for the schedule and sample-order audit.
