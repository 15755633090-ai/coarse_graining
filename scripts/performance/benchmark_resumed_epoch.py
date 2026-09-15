"""Check the remainder of a real resumed Lipo epoch in temporary copies.

Includes cold data-cache preparation and skipped batches; this is a resume
diagnostic, not a steady-state full-epoch performance benchmark.
Never writes formal results or evaluates test samples.
"""
import copy
import shutil
import tempfile
import time
from pathlib import Path

import torch

import run_lipo_formal as entry
from coarse_gnn import TopologyCache
from coarse_gnn.prepared_data import PreparedPropertyDataset, prepared_tools


class EpochFinished(Exception):
    pass


def main():
    torch.set_num_threads(2)
    legacy, settings, _, _ = entry.load_protocol(entry.PROJECT)
    directory = entry.PROJECT / "model/results_formal/05_coarse_gnn/lipo/region_only/tuning/seed_42/trial_0"
    resume = directory / "resume.pt"
    identity = legacy.file_identity(resume)
    state = torch.load(resume, map_location="cpu", weights_only=False)
    start_epoch, start_batch = state["epoch"], state["next_batch"]
    del state
    config = entry.read_json(directory / "run_config.json")
    for key, value in config["selected_hyperparameters"].items():
        setattr(settings, key, value)
    settings.tuning_protocol = dict(sha256=config["tuning_protocol_sha256"], payload=config["tuning_protocol"])
    source, splits, spec = legacy.build_property_data(settings.data_root, "lipo")
    cache = TopologyCache(entry.PROJECT / "model/results_formal/05_coarse_gnn/_topology_cache", max_memory_entries=8192)
    factory = entry.make_factory(legacy.create_property_model, cache)
    report = dict(diagnostic_only=True, test_evaluated=False, original_resume=identity,
                  start_epoch=start_epoch, start_batch=start_batch, results={})
    original_save = legacy._save_downstream_resume
    for packed in (False, True):
        label = "packed" if packed else "serial"
        tools = prepared_tools(legacy, cache, entry.STRUCTURE) if packed else {}
        dataset = PreparedPropertyDataset(source, legacy.collate_property_batch, cache, entry.STRUCTURE) if packed else source
        def stop_after_epoch(path, checkpoint):
            original_save(path, checkpoint)
            if checkpoint["epoch"] > start_epoch and checkpoint["next_batch"] == 0:
                report["results"][label] = dict(last_epoch=checkpoint["history"][-1],
                                                seconds=time.perf_counter() - started)
                raise EpochFinished()
        with tempfile.TemporaryDirectory(prefix="coarse_epoch_") as temporary:
            run = Path(temporary)
            shutil.copy2(resume, run / "resume.pt")
            shutil.copy2(directory / "run_config.json", run / "run_config.json")
            started = time.perf_counter()
            with entry.overrides(legacy, create_property_model=factory,
                                 _save_downstream_resume=stop_after_epoch, **tools):
                try:
                    legacy.run_single(copy.copy(settings), dataset, splits, spec, "region_only", 42,
                                      torch.device(settings.device), run, evaluate_test=False, role="tuning")
                except EpochFinished:
                    pass
                else:
                    raise RuntimeError("Expected to stop after exactly one diagnostic epoch")
        print(label, report["results"][label], flush=True)
    report["speedup"] = report["results"]["serial"]["seconds"] / report["results"]["packed"]["seconds"]
    report["original_resume_unchanged"] = legacy.file_identity(resume) == identity
    if not report["original_resume_unchanged"]:
        raise RuntimeError("Original checkpoint changed during isolated audit")
    entry.write_json(entry.ROOT / "outputs/performance/resumed_epoch.json", report)
    print("epoch speedup", report["speedup"], flush=True)


if __name__ == "__main__":
    main()
