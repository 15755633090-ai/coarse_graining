# Lipo seed-0 supervised fine-tuning (invalidated checkpoint selection)

## Status

These runs completed, but they are **development diagnostics, not valid final
comparisons**. A checkpoint-selection bug compared each validation score with
the immediately preceding epoch instead of the global best. Consequently,
`best.pt` reports the correct global-best epoch from history but contains model
weights from the last local validation decrease.

The histories remain valid. The global-best weights were not retained and
cannot be reconstructed, so both arms must be rerun from the original
`diffusion_encoder/encoder.pt` after commit `4533a04` plus the selection fix.

## Valid validation-history minima

| Arm | Epochs | Global-best epoch | Val RMSE | Val MAE | Val R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| FT Diffusion baseline | 100 | 85 | 0.717291 | 0.537160 | 0.652866 |
| FT Diffusion + Multiscale | 100 | 58 | 0.829982 | 0.624972 | 0.535225 |

On validation, multiscale was worse by `0.112691` RMSE (15.71%) in this run.

## Saved-checkpoint diagnostics (not validation-best results)

Both saved model states came from epoch 97, not from the global-best epochs.
Their one-time test evaluations were:

| Arm | Saved-state epoch | Saved-state Val RMSE | Test RMSE | Test MAE | Test R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| FT Diffusion baseline | 97 | 0.730638 | 0.759189 | 0.572548 | 0.522572 |
| FT Diffusion + Multiscale | 97 | 0.839998 | 0.802281 | 0.624394 | 0.466834 |

These test numbers must not be labeled as results from the global
validation-best checkpoints. Their diagnostic difference is
`RMSE(baseline) - RMSE(multiscale) = -0.043093`.

## Required rerun

Use new output directories and do not resume these invalidated runs. Test must
again be evaluated only once after global validation selection.

