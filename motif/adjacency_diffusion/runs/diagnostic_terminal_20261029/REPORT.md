# Fixed-batch adjacency diagnostics

Same tensor initialization is not matched A_hat initialization. No generalization or method superiority claim.

| lr | t | head | initial ratio | final fixed ratio | fixed F1 | fixed AP | fresh ratio | heldout ratio | overfit |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 3e-05 | 50 | epsilon | 0.311 | 0.2455 | 1 | 1 | 0.2919 | 0.2832 | False |
| 3e-05 | 50 | direct_A | 1.08 | 0.4907 | 0.4603 | 0.8187 | 0.5194 | 0.615 | False |
| 3e-05 | 500 | epsilon | 119.5 | 52.16 | 0.2229 | 0.1439 | 74.78 | 73.07 | False |
| 3e-05 | 500 | direct_A | 1.081 | 0.5339 | 0.4803 | 0.7047 | 0.8073 | 0.9416 | False |
| 3e-05 | 1000 | epsilon | 2.523e+05 | 1.104e+05 | 0.1704 | 0.1028 | 1.556e+05 | 1.517e+05 | False |
| 3e-05 | 1000 | direct_A | 1.008 | 0.4596 | 0.6288 | 0.8061 | 0.8736 | 1.02 | False |
| 0.001 | 50 | epsilon | 0.311 | 0.02151 | 1 | 1 | 0.3473 | 0.3338 | True |
| 0.001 | 50 | direct_A | 1.08 | 0.02779 | 1 | 1 | 0.2453 | 0.2762 | True |
| 0.001 | 500 | epsilon | 119.5 | 0.6469 | 0.9298 | 0.9993 | 1.983 | 2.04 | False |
| 0.001 | 500 | direct_A | 1.081 | 0.002688 | 1 | 1 | 0.9503 | 1.02 | True |
| 0.001 | 1000 | epsilon | 2.523e+05 | 810.1 | 0.1766 | 0.1101 | 842.2 | 803 | False |
| 0.001 | 1000 | direct_A | 1.008 | 0.005275 | 1 | 1 | 0.9896 | 1.048 | True |
