# Fixed-weight path dependency screen

No training; no claim of structural redundancy. Positive RMSE change means removing/bypassing the path hurts.

| arm | intervention | all: mean ΔRMSE | multi-fragment: mean ΔRMSE | low-dependency candidate on multi-fragment |
|---|---|---:|---:|---|
| scratch | full | +0.00000 | 0.0 | False |
| scratch | no_internal_branch | +2.15403 | 1.3111158682499664 | False |
| scratch | no_context_within | +2.50992 | 1.5012982149680265 | False |
| scratch | no_context_cross | +0.07238 | 0.1439982107071137 | False |
| scratch | no_internal_readout | +1.64977 | 0.7229038384749584 | False |
| scratch | no_context_readout | +0.73559 | 0.6140440674310521 | False |
| scratch | uniform_boundary | +0.00856 | 0.017600773941960135 | False |
| scratch | no_internal_and_context_within | +4.79387 | 4.012836818886635 | False |
| masked | full | +0.00000 | 0.0 | False |
| masked | no_internal_branch | +1.58327 | 0.9834619664698593 | False |
| masked | no_context_within | +1.60951 | 0.9587148126123479 | False |
| masked | no_context_cross | +0.01046 | 0.021958981281309724 | False |
| masked | no_internal_readout | +0.70358 | 0.23927075564398484 | False |
| masked | no_context_readout | +0.80552 | 0.6442220366959499 | False |
| masked | uniform_boundary | +0.03700 | 0.07712961742974363 | False |
| masked | no_internal_and_context_within | +1.56936 | 1.0855603215087686 | False |

Joint intervention tests compensation between independent internal and context-within propagation. Direct internal-readout suppression retains internal-to-context/gate routes. Uniform boundary changes both conditionality and message amplitude.
A low-dependency candidate requires per-seed |ΔRMSE|<=0.01 and prediction RMS shift<=0.05. These are descriptive thresholds, not equivalence tests. Fixed-weight suppression is out-of-distribution and is not a retrained removal comparison.
