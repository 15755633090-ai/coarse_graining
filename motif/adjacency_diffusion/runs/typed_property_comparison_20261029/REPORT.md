# Locked three-arm ESOL validation

| task seed | pretraining seed | arm | best epoch | validation RMSE |
|---:|---:|---|---:|---:|
| 0 | None | scratch | 58 | 0.81457 |
| 0 | 7 | full | 18 | 0.88772 |
| 0 | 7 | bounded | 36 | 0.85048 |
| 1 | None | scratch | 52 | 0.77957 |
| 1 | 1 | full | 20 | 0.86252 |
| 1 | 1 | bounded | 25 | 0.84689 |
| 2 | None | scratch | 12 | 0.85560 |
| 2 | 2 | full | 12 | 0.83753 |
| 2 | 2 | bounded | 12 | 0.87105 |

All nine runs completed; paired initialization, checkpoint replay and source hashes passed.

scratch: n=3, mean RMSE=0.81658, sample SD=0.03805594703547176
full: n=3, mean RMSE=0.86259, sample SD=0.02509732538508208
bounded: n=3, mean RMSE=0.85614, sample SD=0.013034455905531067
