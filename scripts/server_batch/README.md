# Server batch readout experiment

This package is intentionally separate from the formal local experiment.
It creates a portable input bundle and launches six independent validation-only
readout jobs: `region_size_weighted` and `base_size_weighted`, each for seeds
0, 1, and 2.  Each job has its own output directory, so concurrent processes
never write the same checkpoint, progress file, or manifest.

The bundle contains only the Lipo inputs, the original downstream source,
selected frozen hyperparameters, checkpoint, frozen feature cache, and the
hashes needed to verify them. It does not copy the 13+ GB historical results
tree or use test metrics.

Run `python -m scripts.server_batch.bundle --destination <directory>` on the
source machine, copy the resulting directory to the compute server, and then
run `python -m scripts.server_batch.launch --bundle <directory> --output-dir <directory> --gpus 0 1`.
