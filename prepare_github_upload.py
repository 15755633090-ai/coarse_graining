"""Create a small, browser-uploadable copy without changing original files."""
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "github_upload"
ENCODER = Path("diffusion/outputs/ogb_clean/encoder.pt")


def include(path: Path) -> bool:
    if path == ENCODER:
        return True
    if any(part in {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "outputs", "github_upload"} for part in path.parts):
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    if path.parts[:2] == ("diffusion", "datasets") and path.suffix in {".csv", ".jsonl"}:
        return False
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return False
    # Allow only project source, documentation, manifests and requirements.
    return path.suffix in {".py", ".md", ".json"} or path.name in {"requirements.txt", ".gitignore"}


def main():
    if DESTINATION.exists():
        raise SystemExit("github_upload already exists. Rename it before preparing a new copy; existing files will not be overwritten.")
    if not (ROOT / ENCODER).is_file():
        raise SystemExit(f"Missing required encoder: {ENCODER}")
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and not path.is_symlink() and include(path.relative_to(ROOT)))
    for source in files:
        target = DESTINATION / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    size = sum(path.stat().st_size for path in files)
    largest = max(path.stat().st_size for path in files)
    print(f"Prepared github_upload: {len(files)} files, {size / 1024**2:.2f} MiB total, {largest / 1024**2:.2f} MiB largest file.")
    print("Upload the contents of github_upload, preserving its subfolders. Original files are unchanged.")


if __name__ == "__main__":
    main()
