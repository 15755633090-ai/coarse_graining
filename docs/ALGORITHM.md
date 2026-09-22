# Topology-centered near-fine/far-coarse tokenization

This document is the implementation contract for the first version of the
model. The molecular graph is an undirected, unweighted simple graph
`G = (V, E)`.

## 1. Diffusion representation

The full molecular graph is passed through the frozen four-layer diffusion
encoder. The encoder returns one hidden state per atom at every layer:

```text
H^(1), H^(2), H^(3), H^(4),    h_v^(l) in R^d.
```

No learned graph propagation is added after tokenization. The four scales are
formed by graph-radius partitioning and by applying a separate token MLP to
each scale.

## 2. Center window and outer scales

The graph-center candidate set is

```text
C_G = argmin_v max_u d_G(u, v).
```

One candidate is sampled uniformly. Let `c_0` be the sampled center. The
center window is

```text
Q = {v in V : d_G(v, c_0) <= 3},    I_1 = Q.
```

Every node in `Q` is an atom-level token using `h_v^(1)`. The readout is

```text
z_Q = (1 / |Q|) sum_{v in Q} h_v^(1).
```

For levels `l = 2, 3, 4`, the standard token radius is

```text
(r_2, r_3, r_4) = (1, 2, 3),
d_l = r_l + 1.
```

At level `l`, the input to every token is the corresponding `h_v^(l)`.

## 3. One outer covering round

Fix an internal set `I`. The unassigned set is `U = V \ I`. Consider a
connected component `C` of `G[U]` that meets the outside boundary:

```text
C intersect partial^+ I != empty.
```

Its depth relative to `I` is

```text
D(C; I) = max_{v in C} d_G(v, I).
```

If `D(C; I) < d_l`, component `C` cannot contain a legal center and is emitted
as one residual token:

```text
R_rem^(l) = C.
```

If `D(C; I) >= d_l`, the component must continue with standard tokens. A legal
center is sampled uniformly from

```text
{c in C : d_G(c, I) = d_l}.
```

After every emitted token, `U` is updated and the connected components of
`G[U]` are recomputed. A standard region is

```text
U_k = V \ (I union union_{j<k} R_j),
R_k = {v in U_k : d_{G[U_k]}(v, c_k) <= r_l}.
```

The round stops as soon as the outside boundary of the fixed `I` is covered.
Residual decisions are always local to one connected component of `G[U]`;
different components are never merged into one residual token.

For `l = 2` and `l = 3`, one round is used:

```text
I_l = I_{l-1} union union_k R_k^(l).
```

For `l = 4`, rounds are repeated:

```text
I_{4,0} = I_3,
I_{4,t} = I_{4,t-1} union union_k R_{k,t}^(4),
```

until `I_{4,t} = V`. Every region produced in any level-4 round belongs to
`z_4` and participates in the same within-scale normalization.

## 4. Token readout

For a region `R_k^(l)` with `n_k^(l) = |R_k^(l)|`,

```text
T_k^(l) = (1 / n_k^(l)) sum_{v in R_k^(l)} h_v^(l),
Y_k^(l) = phi_l(T_k^(l)),
w_k^(l) = n_k^(l) / sum_j n_j^(l),
z_l = sum_k w_k^(l) Y_k^(l).
```

`phi_l` is a separate two-layer GELU MLP for every scale. If a scale is empty,
its representation is exactly zero.

The first-version graph representation and prediction are fixed as

```text
z_G = z_Q + z_2 + z_3 + z_4,
y_hat = f_pred(z_G).
```

No LayerNorm, learned scale coefficient, concatenation projection, token edge,
or post-tokenization GNN is part of the first-version main method.

## 5. Partition randomness

The partition is written `P(G; omega)`. It is sampled using a partition seed
that is separate from model initialization and data-loader seeds.

Training:

```text
Each epoch derives a new partition seed from
(base_partition_seed, sample_id, epoch).
```

Validation and test:

```text
Use epoch = 0 for every sample, giving one fixed partition per molecule.
```

The implemented random choices are:

1. uniform selection from graph-center candidates;
2. uniform selection from legal standard-region centers;
3. the effect of center order through sequential region removal.

Candidate priorities are generated from canonical graph ranks, so their order
does not follow input atom IDs. The tests cover random relabelings of both
asymmetric and strongly symmetric graphs, comparing partitions up to graph
isomorphism. Automorphic vertices are interchangeable under the topology-only
definition, so the implementation does not claim a unique representative among
them. Atom and bond labels are not part of this canonicalization. Training
still samples a new partition for every epoch.

Validation and test reports restore regression predictions to the original
target units before computing RMSE, MAE, and R-squared. Classification reports
use per-task ROC-AUC and average precision, averaging only tasks with labels
from both classes.

## 6. Diagnostics retained for later analysis

Partition statistics are emitted by the implementation but do not alter the
model:

```text
graph diameter,
|Q| / |V|,
probability that H^(2), H^(3), H^(4) occur,
tokens and atoms per scale,
residual trigger rate,
level-4 round count,
||z_Q||, ||z_2||, ||z_3||, ||z_4||.
```

These are the agreed statistics, ablation hooks, and risk checks for later
rounds. They are not additional modules in the first-version main structure.
