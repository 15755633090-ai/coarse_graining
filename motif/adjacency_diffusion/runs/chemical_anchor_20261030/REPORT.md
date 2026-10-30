# Fixed chemical-target pretraining result

| task seed | candidate validation RMSE | minus original scratch | best epoch |
|---:|---:|---:|---:|
| 0 | 0.815820 | +0.000843 | 37 |
| 1 | 0.785196 | +0.026304 | 58 |
| 2 | 0.759233 | -0.028226 | 52 |

Candidate: 0.786750 +/- 0.028325 (n=3).
Original scratch: {'n': 3, 'mean_rmse': 0.7871093541578864, 'sample_sd': 0.02804412413028663}
Original expanded edge: {'n': 3, 'mean_rmse': 0.8428769931987412, 'sample_sd': 0.016842317141882456}

Full 60 epochs, same original downstream model and fresh paired task head. All best checkpoints replayed.
Joint change of input and target; not a single-factor ablation, not KPGT reproduction, and not compute matched.
Calculated MolLogP and other chemical pseudo-labels are additional prior knowledge, not measured target labels.
Reused ESOL validation, no test evaluation. No independent-task confirmation. No subsequent recipe search.
