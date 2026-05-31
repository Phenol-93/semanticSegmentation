"""Download an MMSegmentation pretrained config and checkpoint locally.

This script intentionally downloads model files into the project directory so
later inference steps can use local config + checkpoint files without runtime
model downloads.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_MODEL_CONFIG = "pspnet_r50-d8_4xb2-40k_cityscapes-512x1024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly download an MMSegmentation pretrained Cityscapes "
            "config and checkpoint into a local checkpoints directory."
        )
    )
    parser.add_argument(
        "--model-config-name",
        default=DEFAULT_MODEL_CONFIG,
        help="MMSegmentation model config name used by OpenMIM.",
    )
    parser.add_argument(
        "--dest",
        default="checkpoints",
        help="Destination directory for the downloaded config and checkpoint.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root() / path


def snapshot_files(dest: Path) -> set[Path]:
    if not dest.exists():
        return set()
    return {path.resolve() for path in dest.glob("*") if path.is_file()}


def newest_file(files: list[Path], suffix: str) -> Path | None:
    candidates = [path for path in files if path.suffix.lower() == suffix]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def display_path(path: Path) -> str:
    root = project_root()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def find_mim_executable() -> str | None:
    """Prefer the mim command from the active Python environment."""
    python_path = Path(sys.executable).resolve()
    candidates = []
    if python_path.parent.name.lower() == "scripts":
        candidates.append(python_path.parent / "mim.exe")
        candidates.append(python_path.parent / "mim")
    else:
        candidates.append(python_path.parent / "Scripts" / "mim.exe")
        candidates.append(python_path.parent / "Scripts" / "mim")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("mim")


def find_downloaded_pair(dest: Path, before: set[Path]) -> tuple[Path, Path]:
    after = snapshot_files(dest)
    new_files = sorted(after - before)
    search_files = new_files if new_files else sorted(after)

    config_file = newest_file(search_files, ".py")
    checkpoint_file = newest_file(search_files, ".pth")

    missing = []
    if config_file is None:
        missing.append(".py config file")
    if checkpoint_file is None:
        missing.append(".pth checkpoint file")
    if missing:
        raise RuntimeError(
            "Download command finished, but could not find "
            + " and ".join(missing)
            + f" in {dest}."
        )

    return config_file, checkpoint_file


def main() -> int:
    args = parse_args()
    dest = resolve_project_path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    mim_exe = find_mim_executable()
    if not mim_exe:
        print(
            "ERROR: OpenMIM command 'mim' was not found. "
            "Please install openmim in the active OpenMMLab environment, "
            "for example: pip install -U openmim",
            file=sys.stderr,
        )
        return 2

    command = [
        mim_exe,
        "download",
        "mmsegmentation",
        "--config",
        args.model_config_name,
        "--dest",
        str(dest),
    ]
    command_for_record = [
        "mim",
        "download",
        "mmsegmentation",
        "--config",
        args.model_config_name,
        "--dest",
        display_path(dest),
    ]

    before = snapshot_files(dest)
    print("Running OpenMIM download command:")
    print(" ".join(command_for_record))

    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        print(f"ERROR: Failed to start OpenMIM command: {exc}", file=sys.stderr)
        return 2

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(
            "ERROR: OpenMIM download failed. Inference will not continue. "
            f"Exit code: {result.returncode}",
            file=sys.stderr,
        )
        return result.returncode

    try:
        config_file, checkpoint_file = find_downloaded_pair(dest, before)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    model_info = {
        "model_config_name": args.model_config_name,
        "config_file": display_path(config_file),
        "checkpoint_file": display_path(checkpoint_file),
        "download_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "download_command": " ".join(command_for_record),
    }

    info_path = dest / "active_model_info.json"
    info_path.write_text(
        json.dumps(model_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Download complete.")
    print(f"Config file: {model_info['config_file']}")
    print(f"Checkpoint file: {model_info['checkpoint_file']}")
    print(f"Model info: {display_path(info_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
