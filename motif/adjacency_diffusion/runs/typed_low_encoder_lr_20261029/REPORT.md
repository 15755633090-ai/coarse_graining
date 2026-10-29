# Locked three-arm ESOL validation

| task seed | pretraining seed | arm | best epoch | validation RMSE |
|---:|---:|---|---:|---:|
| 0 | None | scratch | 55 | 0.96222 |
| 0 | 7 | bounded | 12 | 1.19656 |
| 1 | None | scratch | 59 | 0.94409 |
| 1 | 1 | bounded | 20 | 0.94607 |
| 2 | None | scratch | 39 | 0.96490 |
| 2 | 2 | bounded | 14 | 0.97028 |

Aggregates are provisional until all planned runs complete.

scratch: n=3, mean RMSE=0.95707, sample SD=0.01131952771920627
bounded: n=3, mean RMSE=1.03764, sample SD=0.1381677330165816
