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
source machine. The generated directory is self-contained: it includes the
required source code, assets, SHA-256 locks, and `run_experiment.cmd`. Copy that
single directory to the compute server and run `run_experiment.cmd`; no Git
checkout or protocol assembly is required on the server. The launcher is a queue: it runs
at most one job per physical GPU and starts the next job only after completion.
Use paths outside the repository for both the copied bundle and job output: each
job verifies that the exact-commit repository is completely clean before it
starts.

The bundle locks `amp`, deterministic mode, micro-batch size, worker count,
CPU thread count, logical CUDA device, and cuBLAS workspace configuration.
Neither `launch` nor `run_job` accepts command-line overrides for these values.
Finally run `python -m scripts.server_batch.collect` to produce the single
validation-only comparison report.
