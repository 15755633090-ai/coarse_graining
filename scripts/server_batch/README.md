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

This is an explicitly **exploratory three-seed** protocol. It is not a
replacement for the locked five-seed formal protocol.

Run `python -m scripts.server_batch.bundle --destination <directory>` on the
source machine. The generated manifest locks the exact Git commit and SHA-256
of every server-side source file. On the compute server, clone the repository,
checkout that manifest commit, copy the bundle, run one `run_job --action
prepare` check, then run `launch --gpus 0 1`. The launcher is a queue: it runs
at most one job per physical GPU and starts the next job only after completion.
Finally run `python -m scripts.server_batch.collect` to produce the single
validation-only comparison report.
