"""Create one self-contained V2 experiment folder for the Windows GPU server.

The generated folder is deliberately independent of Git and of the author's
local drive layout.  Copy that one folder to the server and run
``run_v2.cmd`` there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve()
SOURCE = HERE.parents[2]
PROJECT = SOURCE.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    return {"size": path.stat().st_size, "sha256": sha256(path)}


def copy_file(source: Path, destination: Path, manifest: dict[str, dict[str, object]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    manifest[destination.as_posix()] = identity(destination)


def copy_tree(source: Path, destination: Path, manifest: dict[str, dict[str, object]]) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            copy_file(path, destination / path.relative_to(source), manifest)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def rebase_protocol(selection: dict[str, object]) -> dict[str, object]:
    """Return the selected formal protocol with package-relative input paths."""
    protocol = selection["tuning_protocol"]
    protocol["checkpoint"]["path"] = "../assets/encoder_best.pt"
    protocol["pretrain_smiles"]["path"] = "../assets/ogb_pretrain_clean.csv"
    protocol["source_files"]["downstream_benchmark.py"]["path"] = (
        "../assets/legacy/downstream_benchmark.py"
    )
    for name in ("property_prediction.py", "model.py", "trainer.py"):
        protocol["source_files"][name]["path"] = f"../assets/legacy/bond_diffusion/{name}"
    return selection


def build(destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}

    code = destination / "code"
    for name in ("run_lipo_v2.py", "run_lipo_formal.py", "requirements.txt"):
        copy_file(SOURCE / name, code / name, files)
    copy_tree(SOURCE / "coarse_gnn", code / "coarse_gnn", files)
    copy_tree(SOURCE / "diffusion" / "bond_diffusion", code / "diffusion" / "bond_diffusion", files)
    copy_file(SOURCE / "diffusion" / "requirements.txt", code / "diffusion" / "requirements.txt", files)

    assets = destination / "assets"
    copy_tree(PROJECT / "data" / "ogbg_mollipo", assets / "data" / "ogbg_mollipo", files)
    pretrain = PROJECT / "扩散GNN" / "01_扩散预训练"
    copy_file(pretrain / "outputs" / "ogb_clean" / "best.pt", assets / "encoder_best.pt", files)
    copy_file(pretrain / "datasets" / "ogb_pretrain_clean.csv", assets / "ogb_pretrain_clean.csv", files)
    copy_file(pretrain / "downstream_benchmark.py", assets / "legacy" / "downstream_benchmark.py", files)
    copy_tree(pretrain / "bond_diffusion", assets / "legacy" / "bond_diffusion", files)

    model = destination / "model"
    root_config = json.loads((PROJECT / "model" / "results_formal" / "04_diffusion" / "run_config.json").read_text(encoding="utf-8"))
    root_config["data_root"] = "../assets/data"
    root_config["search_space"]["path"] = "../model/search_spaces.json"
    write_json(model / "results_formal" / "04_diffusion" / "run_config.json", root_config)
    files[(model / "results_formal" / "04_diffusion" / "run_config.json").as_posix()] = identity(model / "results_formal" / "04_diffusion" / "run_config.json")
    copy_file(PROJECT / "model" / "search_spaces.json", model / "search_spaces.json", files)
    selected_path = PROJECT / "model" / "results_formal" / "04_diffusion" / "lipo" / "pretrained_finetune" / "selected_config.json"
    selection = rebase_protocol(json.loads(selected_path.read_text(encoding="utf-8")))
    portable_selection = model / "results_formal" / "04_diffusion" / "lipo" / "pretrained_finetune" / "selected_config.json"
    write_json(portable_selection, selection)
    files[portable_selection.as_posix()] = identity(portable_selection)

    launcher = destination / "run_v2.cmd"
    launcher.write_text(
        "@echo off\r\n"
        "setlocal\r\n"
        "for %%I in (\"%~dp0.\") do set \"PACKAGE_ROOT=%%~fI\"\r\n"
        "set \"CONDA_EXE=D:\\Users\\nieyuhang\\miniconda3\\Scripts\\conda.exe\"\r\n"
        "set \"CONDA_ENV=polyolefin_ml\"\r\n"
        "set \"PYTHONDONTWRITEBYTECODE=1\"\r\n"
        "cd /d \"%PACKAGE_ROOT%\\code\"\r\n"
        "\"%CONDA_EXE%\" run --no-capture-output -n \"%CONDA_ENV%\" python -u run_lipo_v2.py --project-root \"%PACKAGE_ROOT%\" --output-dir \"%PACKAGE_ROOT%\\results\" %*\r\n"
        "exit /b %ERRORLEVEL%\r\n",
        encoding="utf-8",
    )
    files[launcher.as_posix()] = identity(launcher)
    for variant, gpu, label in (
        ("v2_region_only", 0, "region"),
        ("v2_full", 1, "full"),
    ):
        command = destination / f"run_v2_{label}.cmd"
        command.write_text(
            "@echo off\r\n"
            "call \"%~dp0run_v2.cmd\" --action train --seeds 0 1 2 "
            f"--variants {variant} --device cuda:{gpu} --micro-batch-size 32 --num-workers 4 "
            f"--output-dir \"%~dp0results\\{variant}\"\r\n"
            "exit /b %ERRORLEVEL%\r\n",
            encoding="utf-8",
        )
        files[command.as_posix()] = identity(command)
    parallel = destination / "run_v2_parallel.cmd"
    parallel.write_text(
        "@echo off\r\n"
        "echo Starting one isolated V2 task on each A40 GPU.\r\n"
        "start \"V2 Region-only - GPU 0\" cmd /k call \"%~dp0run_v2_region.cmd\"\r\n"
        "start \"V2 Full - GPU 1\" cmd /k call \"%~dp0run_v2_full.cmd\"\r\n"
        "echo Two training windows were opened. Do not close them while training.\r\n"
        "exit /b 0\r\n",
        encoding="utf-8",
    )
    files[parallel.as_posix()] = identity(parallel)
    portable_files = {str(Path(path).relative_to(destination)): value for path, value in files.items()}
    package_id = hashlib.sha256(json.dumps(portable_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    write_json(destination / "manifest.json", {
        "schema_version": 1,
        "experiment": "v2_exclusive_relation_correction",
        "package_id": package_id,
        "files": portable_files,
        "usage": {
            "prepare": "run_v2.cmd --action prepare",
            "smoke": "run_v2.cmd --action smoke --seeds 0",
            "train": "run_v2.cmd --action train --seeds 0 1 2",
            "parallel_train": "run_v2_parallel.cmd",
        },
    })
    print(f"Created portable V2 package: {destination}")
    print(f"Package ID: {package_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    options = parser.parse_args()
    build(options.destination)


if __name__ == "__main__":
    main()
