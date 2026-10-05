# Independent Review Packet: Steiner Tree QUBO Investigation

Prepared: 29 September 2026  
Repository root: `finale/`  
Original report: `491finalreport.tex`

## Instructions to the reviewing model

Please audit this work as a skeptical mathematical and experimental reviewer. Do not assume that the claims in this document are correct merely because tests pass. In particular:

1. Check every zero-set and nonnegativity argument in the Hamiltonians.
2. Try to construct counterexamples to the chordal-cycle proof, connectivity proof, local parent penalty, and safe penalty bound.
3. Compare the implementations against the displayed formulas.
4. Check whether the benchmark metrics measure what they claim.
5. Separate mathematical correctness from empirical solver performance.
6. Identify experimental confounders, weak claims, missing baselines, and novelty concerns.
7. Report each issue with a severity level and a concrete correction.

The most important source files are:

- `491finalreport.tex`: the student's original research report.
- `SteinerTreeProblemQUBO/AlexFowler/steiner_to_bqm_alex.py`: local Fowler baseline.
- `SteinerTreeProblemQUBO/MyFormulation/steiner_to_bqm_hybrid.py`: original binary-depth/big-M implementation.
- `SteinerTreeProblemQUBO/research/steiner_to_bqm_chordal.py`: proposed chordal-ordering prototype.
- `SteinerTreeProblemQUBO/research/benchmark_repairs.py`: corrected pilot benchmark.
- `SteinerTreeProblemQUBO/research/test_chordal.py` and `test_local_parent.py`: chordal-model tests.
- `SteinerTreeProblemQUBO/research/RESEARCH_NOTE.md`: longer derivations and results.
- `SteinerTreeProblemQUBO/research/audit_findings.md`: audit of the original formulation and benchmark.

The other files under `SteinerTreeProblemQUBO/research/` contain alternative encodings, tests, raw JSON results, and figure generation.

## 1. Original task

The student is researching QUBO formulations of the undirected weighted Steiner Tree Problem. The goal is to improve on Fowler's ordering-based formulation while retaining an exact correspondence between zero-penalty states and valid rooted Steiner trees.

The student's proposed formulation replaces Fowler's pairwise ordering variables with binary-encoded vertex depths. It has better asymptotic variable and interaction counts on sparse bounded-degree graphs. In experiments, however, simulated annealing and simulated quantum annealing performed substantially worse on it than on Fowler's formulation, even for instances where the depth model had fewer terms.

The requested investigation was to determine why the binary-depth formulation performs poorly, whether it can be repaired, whether it has a deeper structural problem, and what direction would give the project a defensible result.

## 2. Problem definition and notation

Let

\[
G=(V,E),\qquad |V|=n,\quad |E|=m,
\]

be a connected undirected graph with nonnegative edge costs \(c_{uv}\). Let \(T\subseteq V\) be the terminals and choose a root \(r\in T\).

For every nonroot original edge, both directed selection variables are available:

\[
e_{uv},e_{vu}\in\{0,1\}.
\]

For a root-incident edge \(\{r,v\}\), only \(e_{rv}\) exists. There is no arc into the root.

Let \(A\) be this set of allowed arcs, so

\[
|A|=2m-d_r,
\]

where \(d_r\) is the degree of the root.

The selected-edge objective is

\[
O(e)=
\sum_{\substack{\{u,v\}\in E\\u,v\ne r}}
c_{uv}(e_{uv}+e_{vu})
+
\sum_{\{r,v\}\in E}c_{rv}e_{rv}.
\]

## 3. Fowler's ordering formulation

The local Fowler implementation creates an ordering bit for every pair of nonroot vertices. Fix a naming order on vertices. For \(u<v\),

\[
x_{uv}=1
\iff
u\text{ is earlier than }v.
\]

The naming relation \(u<v\) is fixed data and is not the optimized order.

### 3.1 Complete-order transitivity

For every nonroot triple \(u<v<w\), Fowler uses

\[
T_{uvw}
=x_{uw}+x_{uv}x_{vw}-x_{uv}x_{uw}-x_{uw}x_{vw}.
\]

This polynomial is one for the two cyclic orientations of the triangle and zero for its six acyclic orientations. Thus

\[
F_1(x)=\sum_{u<v<w}T_{uvw}.
\]

### 3.2 Selected-edge/order consistency

For every original nonroot edge \(\{u,v\}\) with \(u<v\),

\[
F_2(e,x)
=\sum_{\{u,v\}\in E}
\left[e_{uv}(1-x_{uv})+e_{vu}x_{uv}\right].
\]

A selected directed edge must point forward in the order. Selecting both directions costs one regardless of \(x_{uv}\).

### 3.3 Terminal parents

For every nonroot terminal,

\[
F_3(e)=
\sum_{v\in T\setminus\{r\}}
\left(1-\sum_{u:(u,v)\in A}e_{uv}\right)^2.
\]

### 3.4 Fowler nonterminal penalty

For a nonterminal \(v\), define selected indegree and outdegree

\[
q_v=\sum_u e_{uv},
\qquad
s_v=\sum_w e_{vw}.
\]

The implementation uses the grouped penalty

\[
F_v^{\mathrm{Fowler}}
=n\binom{q_v}{2}+(1-q_v)s_v.
\]

The first term penalizes multiple parents. The second prevents a nonterminal with no parent from producing children. Although \((1-q_v)s_v\) can be negative when \(q_v>1\), the grouped expression is nonnegative because \(s_v\le n-1\).

Its zero set is:

- \(q_v=0,s_v=0\): unused nonterminal;
- \(q_v=1\), any allowed \(s_v\): used nonterminal with one parent.

A nonterminal with one incoming edge and no outgoing edge is allowed. It is a dangling nonterminal leaf. With strictly positive costs, an optimum removes it; with zero-cost edges, an optimal solution can contain such redundant branches.

### 3.5 Fowler Hamiltonian

The baseline is

\[
H_{\mathrm{Fowler}}
=O(e)+P\left(F_1+F_2+F_3+\sum_{v\in V\setminus T}F_v^{\mathrm{Fowler}}\right).
\]

It has \(\binom{n-1}{2}\) ordering variables and its triangle term produces \(3\binom{n-1}{3}\) ordering interactions before considering the other constraints.

## 4. Original binary-depth formulation

The student's formulation replaces complete pairwise order with binary depths

\[
o_v=\sum_{i=0}^{B-1}2^i o_{v,i},
\qquad B=\lceil\log_2 n\rceil.
\]

For each allowed arc, it encodes

\[
e_{uv}=1\Longrightarrow o_v\ge o_u+1.
\]

With big-M and nonnegative binary slack \(g_{uv}\), the residual is

\[
r_{uv}=o_v-o_u-1+M(1-e_{uv})-g_{uv},
\]

and the penalty is

\[
H_{\mathrm{depth}}=\sum_{(u,v)\in A}r_{uv}^2.
\]

The remaining constraints use nonterminal usage bits \(p_v\):

\[
\left(p_v-\sum_u e_{uv}\right)^2
\]

and

\[
\sum_w e_{vw}(1-p_v).
\]

The mathematical zero set is sound: strict depth increase forbids directed cycles, and the parent/fake-root constraints make every selected component reach the unique root.

## 5. Main diagnosis of the depth model

The problem is stronger than a large maximum matrix coefficient.

At a satisfied arc constraint, \(r_{uv}=0\). If the edge bit flips while all other bits remain fixed, the residual changes by exactly \(\pm M\). Therefore the single-bit move has depth penalty

\[
\boxed{\Delta H_{\mathrm{depth}}=P M^2.}
\]

If depth bit \(b\) of vertex \(v\) flips while all witnesses remain fixed, each incident allowed arc changes its residual by \(\pm2^b\). If \(a_v\) allowed arcs contain \(v\), then

\[
\boxed{\Delta H_{\mathrm{depth}}=P a_v4^b.}
\]

Inactive arcs contribute as well. They do not logically constrain depths after their slack is optimized, but their current slack certificates resist local changes to the depths.

This explains why a model with fewer interactions can still be difficult for a sampler based on single-bit moves. Removing or adding an edge requires coordinated changes to edge, slack, and possibly rank bits, and binary carry transitions require several bits to change.

Global scaling does not change these barriers relative to the constraint gap or objective. It can correct an absolute mismatch with a fixed solver schedule, but it cannot change the Hamiltonian's relative landscape. Scaling only the depth term down by \(M^2\) also scales down its smallest positive violation penalty; restoring the same worst-case exactness guarantee restores the scale separation.

This is not an impossibility proof for all possible algorithms. Cluster moves, repair, tailored schedules, or other dynamics may perform better. It is a structural objection to expecting ordinary single-bit annealing to be repaired through normalization alone.

## 6. Safe penalty bound

For a Hamiltonian whose unweighted constraint function is nonnegative and integer-valued, let \(U\) be the cost of any feasible Steiner tree. Then

\[
P>U
\]

is sufficient for exactness:

- a feasible assignment with energy at most \(U\) exists;
- any infeasible assignment has at least one unit of violation and energy at least \(P>U\).

This is conservative, not necessary.

The frequently suggested condition \(P>c_{\max}\) is not sufficient in general. Consider an \(n\)-vertex path whose endpoints are the only terminals and whose edges all cost \(c\). The feasible optimum costs \((n-1)c\). The empty edge set can violate only the nonroot terminal-parent constraint and have energy \(P\). Hence it beats the feasible optimum whenever \(P<(n-1)c\), even if \(P>c\).

With different weights for separate penalties, setting every weight greater than \(U\) is a simple sufficient condition when each separately weighted component is nonnegative. Fowler's multiple-parent and fake-root expressions must be treated as one grouped nonnegative component unless a new bound is proved.

## 7. Proposed chordal-ordering formulation

The recommended model retains Fowler's ordering mechanism but sparsifies the graph on which it is applied.

### 7.1 Chordal completion

Remove the root and form

\[
G_0=G[V\setminus\{r\}].
\]

Construct a chordal completion

\[
H=(V\setminus\{r\},F),
\qquad E(G_0)\subseteq F.
\]

Every cycle of length at least four in \(H\) has a chord. The prototype uses deterministic greedy minimum-fill vertex elimination: when eliminating a vertex, connect all of its remaining neighbors into a clique, then remove it from the working graph.

Edges in \(F\setminus E\) are ordering scaffolding only. They have no selection variables, no cost, and cannot appear in a Steiner tree.

### 7.2 Variables

The selected-edge variables \(e_{uv}\) are exactly as before and exist only for original edges.

For every completion edge \(\{u,v\}\in F\), with fixed name order \(u<v\), introduce one ordering bit \(x_{uv}\). Thus the model does not create ordering variables for nonedges outside the chordal completion.

### 7.3 Edge/order consistency

For every original nonroot edge, use the same Fowler penalty:

\[
F_{\mathrm{edge}}
=\sum_{\substack{\{u,v\}\in E\\u,v\ne r,\ u<v}}
\left[e_{uv}(1-x_{uv})+e_{vu}x_{uv}\right].
\]

### 7.4 Triangle consistency

For every triangle \(\{u,v,w\}\) of \(H\), label it \(u<v<w\) and use

\[
T_{uvw}
=x_{uw}+x_{uv}x_{vw}-x_{uv}x_{uw}-x_{uw}x_{vw}.
\]

Then

\[
F_{\triangle}=\sum_{\{u,v,w\}\text{ triangle in }H}T_{uvw}.
\]

### 7.5 Parent constraints

The direct comparison version uses the exact same terminal and nonterminal terms as Fowler:

\[
F_{\mathrm{terminal}}
=\sum_{v\in T\setminus\{r\}}(1-q_v)^2,
\]

\[
F_{\mathrm{nonterminal}}
=\sum_{v\in V\setminus T}
\left[n\binom{q_v}{2}+(1-q_v)s_v\right].
\]

The chordal/Fowler-parent Hamiltonian is therefore

\[
\boxed{
H_{\mathrm{chordal}}
=O(e)+P\left(
F_{\triangle}+F_{\mathrm{edge}}+F_{\mathrm{terminal}}+F_{\mathrm{nonterminal}}
\right).
}
\]

Only the ordering graph and the set of triangle penalties differ from Fowler's baseline.

## 8. Correctness argument for chordal ordering

### 8.1 No directed cycle survives the triangle penalties

Orient every edge of \(H\) according to its ordering bit. Suppose this orientation contains a directed cycle and choose a shortest one.

If the cycle has length at least four, chordality supplies a chord between two nonconsecutive cycle vertices. The original directed cycle gives a directed path from the first chord endpoint to the second in each direction, one along each side of the cycle. Whichever way the chord is oriented, it closes one of these paths into a strictly shorter directed cycle. That contradicts minimality.

Therefore every directed cycle in an oriented chordal graph contains a directed triangle. Since all directed triangles are penalized, zero triangle penalty implies that the orientation of \(H\) is acyclic.

Selected original edges agree with this orientation, so selected edges cannot contain a directed cycle either.

### 8.2 No valid tree is excluded

Take any valid Steiner tree and orient it away from the root. The selected nonroot arcs form a DAG, so choose a topological order. Orient every edge of \(H\), including fill edges, according to that order. Every selected edge agrees with the order and every triangle is acyclic. The parent terms also vanish.

Therefore every valid Steiner tree has a zero-penalty extension.

### 8.3 Parent constraints imply one rooted component

At zero penalty:

- every nonroot terminal has one incoming selected edge;
- every used nonterminal has one incoming selected edge;
- a nonterminal without a parent has no children;
- no selected directed cycle exists;
- no edge enters the root.

Follow parent arcs backward from any selected nonroot vertex. The chain cannot cycle. It cannot stop at a nonroot terminal because terminals have parents. It cannot stop at a nonterminal that has a child because such a nonterminal must have a parent. Hence it must reach the root.

Thus all selected edges form one rooted acyclic component spanning every terminal. Redundant nonterminal leaves are allowed, as in Fowler; positive costs remove them from an optimum.

## 9. Why chordal completion is necessary

Using triangle penalties only on the original graph is incorrect. A chordless directed four-cycle has no triangle to penalize.

For a square \(A-B-C-D-A\), add chord \(A-C\). If the original cycle is directed, then:

- orientation \(A\to C\) creates directed triangle \(A\to C\to D\to A\);
- orientation \(C\to A\) creates directed triangle \(A\to B\to C\to A\).

The completion guarantees this phenomenon for all longer cycles.

## 10. Difference from Fowler

Let \(q=|F|\) be the number of chordal-completion edges and \(\tau\) the number of its triangles.

| Component | Fowler | Chordal formulation |
|---|---:|---:|
| Ordering graph | Complete graph on \(V\setminus\{r\}\) | Chordal completion of \(G_0\) |
| Ordering variables | \(\binom{n-1}{2}\) | \(q\) |
| Ordering-triangle interactions | \(3\binom{n-1}{3}\) | \(3\tau\) |
| Edge/order penalty | Same | Same |
| Parent penalty | Can be identical | Can be identical |
| Worst-case complexity | \(O(n^2)\) variables, \(O(n^3)\) interactions | Same |

If the chosen elimination order has width \(w\), meaning at most \(w\) remaining neighbors at each elimination step, then

\[
q=O(nw),
\qquad
\tau=O(nw^2).
\]

For bounded \(w\), the ordering portion is linear in \(n\). In the worst case \(w=O(n)\), the completion can become complete and the model becomes essentially Fowler's ordering model.

Sparsity alone does not imply low width. Sparse expanders can require substantial fill. Claims should therefore be stated in terms of completion size or elimination width, not merely \(m=O(n)\).

The core claim is:

> Chordal ordering preserves Fowler's exact cycle-prevention logic while retaining only the comparisons needed for cycle detection in the completed input graph.

It is an adaptive sparsification, not a fundamentally unrelated formulation.

## 11. Optional local nonterminal coefficient

The prototype also tests a smaller valid replacement for Fowler's global factor \(n\). Let \(d_v^+\) be the number of allowed outgoing arcs from nonterminal \(v\), and set

\[
\alpha_v=d_v^++1.
\]

Use

\[
F_v^{\mathrm{local}}
=\alpha_v\binom{q_v}{2}+(1-q_v)s_v.
\]

For \(q_v=0\), this equals \(s_v\). For \(q_v=1\), it is zero. For \(q_v\ge2\),

\[
F_v^{\mathrm{local}}
=(q_v-1)\left(\frac{\alpha_vq_v}{2}-s_v\right).
\]

Since \(s_v\le d_v^+<\alpha_v\), the value is positive. Thus this grouped QUBO has the same zero set and is nonnegative.

On bounded-degree graphs, this prevents the nonterminal multiple-parent coefficient from growing with \(n\). It is a second change, separate from chordal sparsification. Any clean ablation must compare:

- full ordering/Fowler parents;
- chordal ordering/Fowler parents;
- full ordering/local parents;
- chordal ordering/local parents.

## 12. Other exact alternatives implemented

These models were investigated to distinguish coefficient range from auxiliary coordination.

### 12.1 Bitwise strict-depth comparator

For each allowed arc, unsigned subtraction \(o_u-o_v\) is enforced bit by bit:

\[
o_{u,i}-o_{v,i}-b_i-d_i+2b_{i+1}=0,
\qquad b_0=0.
\]

The final borrow is one exactly when \(o_u<o_v\). The edge gate is

\[
e_{uv}(1-b_B).
\]

Primitive quadratic coefficients are at most \(4P\), with no big-M or binary place values in a penalty. The model is in `steiner_to_bqm_circuit.py`.

### 12.2 Shared binary-order comparator

One comparator is shared between the two directions of each nonroot undirected edge. Equal binary ranks are broken by a fixed vertex index, producing a true lexicographic order. Root arcs require no comparator. The model is in `steiner_to_bqm_binary_order.py`.

### 12.3 Domain-wall depths

Unary threshold variables encode depths, and conditional inequalities are quadratized with nonnegative gadgets. Quadratic coefficient magnitude is at most \(2P\), but the model uses \(O(n^2)\) variables/interactions on sparse bounded-degree graphs. It performed poorly in this pilot. The model is in `steiner_to_bqm_domain_wall.py`.

These alternatives show that the big coefficients can be removed exactly. Their weaker sampling results also show that coefficient magnitude is not the only issue: interaction degree and coordinated witness updates matter.

## 13. Benchmark methodology

The new pilot benchmark is `research/benchmark_repairs.py`.

### 13.1 Instances

Two synthetic graph families were used:

- ladder graphs;
- connected random 3-regular graphs.

Edge costs are random positive multiples of \(1/20\). The root is vertex `0`; the remaining terminals are reproducibly sampled.

Small pilot:

- \(n\in\{8,12,16\}\);
- three graph seeds per size and family;
- 18 instances total;
- 100 reads and 2,000 sweeps per model/instance.

Larger pilot:

- \(n\in\{24,32\}\);
- two graph seeds per size and family;
- 8 instances total;
- 100 reads and 5,000 sweeps per model/instance.

### 13.2 Exact reference values

For smaller instances, the optimum is obtained by enumerating every subset of optional vertices and computing an MST on each connected induced subgraph. This is exact because every Steiner tree has some vertex set and is at least the MST of that set, while every connected set's MST is a feasible tree.

For the 32-vertex cases, the existing Gurobi flow ILP provides the certified optimum.

### 13.3 Penalty choice

A feasible upper bound \(U\) is obtained by computing a spanning tree and repeatedly removing nonterminal leaves. Every compared model uses

\[
P=1.05U.
\]

The known optimum is not used to set \(P\).

### 13.4 Metrics

Every selected directed-edge sample is decoded independently of its auxiliary bits.

- **Graph-feasible:** selected arcs form a rooted arborescence spanning all terminals, with no disconnected selected component.
- **Optimal:** graph-feasible and actual selected-edge cost equals the exact optimum.
- **Zero penalty:** full BQM energy minus selected directed-edge cost is zero within tolerance.

All models receive the full read budget. The primary pilot uses `neal.SimulatedAnnealingSampler` with automatic temperature scheduling and deterministic seeds.

The benchmark is exploratory. Model refinements were chosen after seeing initial results, so the aggregate is not an untouched test set and should not support formal significance claims.

## 14. Pilot results

### 14.1 Small instances

| Model | Instances with at least one optimum | Feasible samples | Optimal samples | Zero-penalty samples |
|---|---:|---:|---:|---:|
| Original big-M depth, corrected common weight | 0/18 | 0.3% | 0.0% | 0.1% |
| Full Fowler ordering/Fowler parents | 15/18 | 41.4% | 33.3% | 41.4% |
| Full ordering/usage-bit parents | 12/18 | 41.5% | 19.3% | 41.5% |
| Bitwise strict-depth circuit | 10/18 | 13.9% | 9.9% | 3.5% |
| Shared binary-order comparator | 12/18 | 25.1% | 13.9% | 16.2% |
| Domain-wall depth | 4/18 | 3.9% | 3.8% | 3.9% |
| Chordal ordering/usage-bit parents | 15/18 | 67.2% | 24.3% | 67.2% |
| Chordal ordering/Fowler parents | 17/18 | 65.8% | 38.3% | 65.8% |
| Chordal ordering/local parents | 16/18 | 66.8% | 39.0% | 66.8% |
| Full ordering/local parents | 15/18 | 39.4% | 30.5% | 39.4% |

The cleanest isolation of chordal sparsification is full Fowler versus chordal/Fowler: 41.4% versus 65.8% feasible samples and 33.3% versus 38.3% optimal samples. This is promising exploratory evidence, not a universal performance result.

### 14.2 Larger instances

| Model | Instances with at least one optimum | Feasible samples | Optimal samples |
|---|---:|---:|---:|
| Big-M depth | 0/8 | 0.0% | 0.0% |
| Full Fowler | 0/8 | 10.2% | 0.0% |
| Shared comparator | 0/8 | 0.6% | 0.0% |
| Chordal/usage-bit parents | 0/8 | 66.0% | 0.0% |
| Chordal/Fowler parents | 2/8 | 52.1% | 2.1% |
| Chordal/local parents | 3/8 | 50.4% | 2.8% |

The main positive result is feasibility and model-size improvement on these families. Optimal sampling remains difficult.

### 14.3 Representative 32-vertex ladder

| Model | Variables | Nonzero interactions | Largest quadratic magnitude divided by \(P\) |
|---|---:|---:|---:|
| Big-M depth | 812 | 9,874 | 2,048 |
| Full Fowler | 555 | 13,825 | 32 |
| Shared comparator | 707 | 2,326 | 4 |
| Chordal/usage-bit parents | 170 | 386 | 2 |
| Chordal/Fowler parents | 148 | 424 | 32 |
| Chordal/local parents | 148 | 424 | 4 |

This case is informative because the big-M model already has fewer interactions than full Fowler, yet samples much worse. Chordal/local reduces full Fowler interactions by about 32.6 times while keeping small coefficients.

## 15. OpenJij cross-check and seed issue

A four-instance OpenJij SA cross-check used 100 reads, 2,000 sweeps, and a distinct deterministic seed for each read.

| Model | Instances solved | Feasible samples | Optimal samples |
|---|---:|---:|---:|
| Big-M | 0/4 | 0.0% | 0.0% |
| Full Fowler | 4/4 | 81.75% | 74.5% |
| Shared comparator | 4/4 | 49.25% | 22.25% |
| Chordal/local | 4/4 | 85.25% | 74.0% |

This small check supports comparable full/chordal ordering performance, not clear chordal superiority.

The installed OpenJij 0.10.17 reuses a supplied seed for each read in a multi-read batch. The project's newer notebook passes `seed=0` with many reads. A focused diagnostic obtained one distinct state from eight fixed-seed reads, versus eight distinct states when each read received its own seed, for both SA and SQA. See `diagnose_openjij_seed.py` and `openjij_seed_diagnostic.json`.

This does not affect the Neal pilots. It does mean that the newer seeded OpenJij notebook may repeat one trajectory and should not treat its reads as independent exploration. It does not establish that every historical unseeded benchmark had this issue.

## 16. Problems found in the original benchmark/report

These findings should be independently verified against the current checkout and, if available, the historical commit that generated the report figures.

1. The old benchmark uses total penalized Hamiltonian energy as “cost,” rather than decoded selected-edge cost.
2. It updates the best energy before checking graph feasibility, so an infeasible low-energy state can be classified as an optimum hit.
3. `SampleSet.data()` is energy-sorted by default; enumeration order is not chronological read order, so the reported “first-hit read” is not a true first-hit index.
4. Stopping future batches after a hit gives successful models smaller, outcome-correlated budgets when computing feasibility rates.
5. Several wrappers refer to `MyFormulization` rather than `MyFormulation` and use outdated signatures. The current checkout cannot reproduce them unchanged.
6. The current depth builder's module defaults are all one, whereas Fowler has a cost-dependent default. However, the newer comparison notebook explicitly overrides depth-model weights and normalizes costs, so the module defaults must not be assumed to describe that notebook's actual experiment.
7. OpenJij SQA defaults use an absolute schedule that is not automatically rescaled to each Hamiltonian's magnitude. Equal sweeps do not imply comparable effective temperature/transverse-field scaling.
8. The report's exact depth-model variable count uses \(B\) slack bits per arc, while the implementation uses \(B+1\). The implementation count is

   \[
   |A|(B+2)+nB+(n-|T|).
   \]

   The per-square interaction expression similarly needs \(3B+2\) variables rather than \(3B+1\), and summing clique contributions can double-count shared assembled interactions. The asymptotic upper bound remains valid.

These issues do not by themselves invalidate the observed qualitative difficulty of the big-M model. They mean the quantitative historical metrics should be regenerated before publication.

## 17. Tests performed

The test suite currently contains 23 passing tests. It covers:

- bitwise subtractor truth tables;
- exact borrow comparison over all small bit widths;
- local energy reconstruction for random assignments;
- exact global minima on tiny instances;
- domain-wall quadratization and cycle rejection;
- all orientations of a chordally completed four-cycle;
- chordal completion/perfect-elimination properties on random graphs;
- valid-tree zero-penalty extension;
- triangle coefficient/Ising-field identities;
- equality of full-completion/Fowler-parent mode to the local Fowler implementation on a small instance;
- nonnegativity and exact zero set of the local nonterminal penalty.

Reproduce with:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover \
  -s SteinerTreeProblemQUBO/research -p 'test_*.py'
```

Observed on 29 September 2026:

```text
.......................
----------------------------------------------------------------------
Ran 23 tests in 0.504s

OK
```

Tests provide evidence that the implementation matches the intended local identities. They are not a proof of annealing performance, novelty, or generalization.

## 18. Reproduction commands

From the repository root:

```sh
# Primary small pilot
PYTHONDONTWRITEBYTECODE=1 venv/bin/python \
  -m SteinerTreeProblemQUBO.research.benchmark_repairs \
  --reads 100 --sweeps 2000 --seeds 3 --sizes 8 12 16

# Refinement/parent ablations
PYTHONDONTWRITEBYTECODE=1 venv/bin/python \
  -m SteinerTreeProblemQUBO.research.benchmark_repairs \
  --reads 100 --sweeps 2000 --seeds 3 --sizes 8 12 16 \
  --models binary_order chordal_fowler chordal_local full_order_local \
  --output SteinerTreeProblemQUBO/research/refinement_results.json

# Larger pilot; 32-vertex reference branch requires Gurobi
PYTHONDONTWRITEBYTECODE=1 venv/bin/python \
  -m SteinerTreeProblemQUBO.research.benchmark_repairs \
  --reads 100 --sweeps 5000 --seeds 2 --sizes 24 32 \
  --models big_m fowler binary_order chordal chordal_fowler chordal_local \
  --output SteinerTreeProblemQUBO/research/larger_results.json

# OpenJij SA cross-check with distinct per-read seeds
PYTHONDONTWRITEBYTECODE=1 venv/bin/python \
  -m SteinerTreeProblemQUBO.research.benchmark_repairs \
  --reads 100 --sweeps 2000 --seeds 1 --sizes 8 12 \
  --models big_m fowler binary_order chordal_local --sampler openjij_sa \
  --output SteinerTreeProblemQUBO/research/openjij_sa_results.json

# Summaries and figure
PYTHONDONTWRITEBYTECODE=1 venv/bin/python \
  -m SteinerTreeProblemQUBO.research.summarize_repairs

# Focused OpenJij seed diagnostic
PYTHONDONTWRITEBYTECODE=1 venv/bin/python \
  -m SteinerTreeProblemQUBO.research.diagnose_openjij_seed
```

Saved results:

- `pilot_results.json`
- `refinement_results.json`
- `larger_results.json`
- `openjij_sa_results.json`
- `openjij_seed_diagnostic.json`
- `summary.json`
- `scaling_results.json`
- `comparison.png`
- `environment.json`

## 19. Known limitations and claims that should not be made yet

1. The graphs are synthetic, not SteinLib or real network instances.
2. The instance count is small and refinement choices were data-informed.
3. No confidence intervals or held-out final test set have been produced.
4. No quantum hardware experiment has been performed.
5. The only new OpenJij comparison is a four-instance SA check; broad SQA comparison remains undone.
6. Low-width graph classes can also be classically easier. Compactness on them is a formulation result, not a quantum advantage claim.
7. A greedy completion is used. Its measured width is an upper bound and need not equal treewidth.
8. Sparse graphs can have high width, so “all sparse graphs” is too broad.
9. Fowler's redundant complete ordering may occasionally aid optimization despite its size. Smaller does not imply uniformly easier.
10. The chordal model's worst case is the same \(O(n^2)\)-variable, \(O(n^3)\)-interaction behavior as Fowler.
11. The exact novelty of this Steiner-specific QUBO has not been established by a systematic literature review.
12. The main report has not yet been rewritten to incorporate these results.

## 20. Relevant prior work and novelty boundary

- Fowler, *Improved QUBO Formulations for D-Wave Quantum Computing* (2017): direct ordering baseline. The local code implements this formulation.
- Rankooh and Rintanen, *Propositional Encodings of Acyclicity and Reachability by Using Vertex Elimination*, AAAI 2022: https://ojs.aaai.org/index.php/AAAI/article/view/20530. This is directly relevant prior art for using elimination graphs to sparsify acyclicity encodings. The present chordal QUBO should be framed as a Steiner/QUBO adaptation, not as the invention of elimination-based acyclicity.
- Chancellor, *Domain wall encoding of discrete variables for quantum annealing and QAOA* (2019): https://arxiv.org/abs/1903.05068.
- Chen, Stollenwerk and Chancellor, *Performance of Domain-Wall Encoding for Quantum Annealing* (2021): https://arxiv.org/abs/2102.12224.
- Karimi and Ronagh, *Practical Integer-to-Binary Mapping for Quantum Annealers*: https://arxiv.org/abs/1706.01945.
- Zaman, Tanahashi and Tanaka, *PyQUBO*: https://arxiv.org/abs/2103.01708. Arithmetic QUBO circuits are standard; novelty should not be claimed for binary subtractor gates themselves.
- Liu and Dinneen, *Solving the Bounded-Depth Steiner Tree Problem using an Adiabatic Quantum Computer* (2019): https://doi.org/10.1109/CSDE48274.2019.9162395. This should be reviewed before making broad novelty claims about depth-based Steiner formulations.

The targeted search conducted during this work did not locate the exact chordal-completion Steiner QUBO presented here. That is not proof that it is novel.

## 21. Questions the independent reviewer should answer

Please explicitly answer the following:

1. Is the original depth formulation's zero set exactly the set of rooted Steiner trees, up to redundant positive-cost leaves and auxiliary degeneracy?
2. Is the \(PM^2\) single-edge-flip barrier derivation correct and relevant to single-bit annealing?
3. Does every cyclic orientation of an undirected chordal graph contain a directed triangle? Is the proof above complete?
4. Does every valid selected Steiner tree extend to a zero-penalty orientation of every chordal completion?
5. Do the parent constraints plus acyclicity force every selected component to reach the root?
6. Is the local parent coefficient \(d_v^++1\) sufficient for all possible \(q_v,s_v\)?
7. Does `steiner_to_bqm_chordal.py` implement the displayed Hamiltonian exactly for all three parent modes?
8. Does `full=True, parent_encoding="fowler"` reproduce Fowler's local implementation, apart from irrelevant zero terms or labeling?
9. Are the claimed variable/interaction bounds correct? Distinguish generated term contributions from unique assembled BQM interactions.
10. Is \(P>U\) sufficient under every implemented parent mode, and are all grouped constraint functions nonnegative integers?
11. Are the exact reference solvers and decoding rules correct?
12. Are any benchmark comparisons unfair because of schedules, random seeds, variable counts, coefficient normalization, or tuning?
13. Do the JSON files support every number quoted in this packet?
14. Which conclusions are mathematically established, which are empirical, and which remain hypotheses?
15. What is the strongest defensible research claim, and what experiments are still necessary?

## 22. Current recommended conclusion

The depth formulation appears mathematically valid but locally hostile to single-bit annealing because of its big-M slack certificates. Bitwise encodings eliminate coefficient growth but did not fully repair sampling, indicating that auxiliary coordination also matters.

Chordal ordering is the most promising direction found. It preserves Fowler's exact cycle-prevention structure, has the same worst-case complexity, and can be dramatically smaller on graphs with low elimination width. The pilot results show improved feasibility and competitive optimal sampling on the tested synthetic families.

The responsible current claim is therefore:

> A chordal-completion sparsification of Fowler's ordering formulation is an exact Steiner-tree QUBO. Its size is controlled by elimination width, it reduces to Fowler in the worst case, and it showed promising feasibility and model-size improvements in preliminary synthetic experiments.

It is not yet justified to claim universal solver superiority, practical quantum advantage, or novelty without a broader literature search and held-out benchmarking.
