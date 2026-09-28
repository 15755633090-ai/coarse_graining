# Center-token design audit before version 2 training

All published Lipo and ESOL train/valid/test molecules were read from the
local datasets. Frozen H2/H3/H4 came from the respective validation-best
Stage-1 checkpoint in FP32 evaluation mode. This is a descriptive audit; it
does not train or evaluate the center-token model.

| Dataset | Molecules | `K=8` under old cap | Old cap actually truncated (`N>32`) | New cap 32 truncated (`N>128`) | Maximum N |
|---|---:|---:|---:|---:|---:|
| Lipo | 4,200 | 1,854 (44.14%) | 1,013 (24.12%) | 0 | 115 |
| ESOL | 1,128 | 27 (2.39%) | 10 (0.89%) | 0 | 55 |

The old budget was `min(N, 8, max(3, ceil(N/4)))`. Molecules with N=29–32
naturally receive 8 centers; the cap actually truncates only N>32. With the
new safety cap of 32, the largest observed molecule receives 29 centers at
stride 4.

| Dataset | Atoms assessed per layer | Minimum H2 norm | Minimum H3 norm | Minimum H4 norm | Norms below 1e-6 |
|---|---:|---:|---:|---:|---:|
| Lipo | 113,568 | 10.98154 | 9.91391 | 4.58692 | 0 in every layer |
| ESOL | 14,991 | 11.14452 | 9.96615 | 6.55286 | 0 in every layer |

Zero-length embeddings do not occur in these data. Version 2 still assigns a
near-zero candidate the lowest feature-novelty rank, so future datasets have
defined behavior at this boundary.

Version 2 keeps the original `5+` pairwise attention distance bucket. A change
to that encoding would modify a second model component and can be evaluated
separately. The non-overlapping-core and topology-coverage rules are specified
in [`docs/CENTER_TOKENS.md`](../docs/CENTER_TOKENS.md).

## Permutation audit

The first 20 training molecules of each dataset were each randomly renumbered
three times. Before replacing atom-index tie-breaks, 9/60 Lipo and 4/60 ESOL
permutations changed the untrained center-token model's prediction by more
than 1e-5; maximum absolute changes were 0.08430 and 0.09747. The changed
core indices explained these differences. After resolving topology ties with
atom/bond/distance signatures, **0/60** permutations in either dataset
exceeded 1e-5. Maximum absolute changes were 4.77e-7 for Lipo and 3.58e-7
for ESOL. This sample does not prove invariance for every molecular graph.
