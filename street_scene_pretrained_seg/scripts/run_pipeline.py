"""Run the complete local-checkpoint street-scene segmentation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local MMSegmentation inference, project-class remapping, "
            "visualization, and class statistics. This pipeline does not train "
            "a model and does not require ground-truth masks."
        )
    )
    parser.add_argument("--input-dir", default="input_images", help="Input image directory.")
    parser.add_argument("--out-dir", default="outputs", help="Output root directory.")
    parser.add_argument(
        "--model-info",
        default="checkpoints/active_model_info.json",
        help="Model info JSON containing local config_file and checkpoint_file.",
    )
    parser.add_argument("--config-file", default=None, help="Explicit local MMSeg config file.")
    parser.add_argument("--checkpoint-file", default=None, help="Explicit local .pth checkpoint file.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, such as cuda:0 or cpu.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha for project visualization.")
    return parser.parse_args()


def resolve_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    return (PROJECT_ROOT / path).resolve()


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== START: {name} ===", flush=True)
    print("Command:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(
            f"ERROR: Step failed: {name}. Exit code: {result.returncode}. "
            "Stopping the pipeline.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(result.returncode)
    print(f"=== DONE: {name} ===", flush=True)


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    out_dir = resolve_path(args.out_dir)
    model_info = resolve_path(args.model_info)
    config_file = resolve_path(args.config_file)
    checkpoint_file = resolve_path(args.checkpoint_file)

    if not model_info.is_file() and not (config_file and checkpoint_file):
        print(
            f"ERROR: Model info file not found: {model_info}\n"
            "Please run scripts/00_download_pretrained_model.py first, or pass "
            "both --config-file and --checkpoint-file.",
            file=sys.stderr,
        )
        return 2

    if bool(config_file) != bool(checkpoint_file):
        print(
            "ERROR: Please pass both --config-file and --checkpoint-file, "
            "or pass neither and use --model-info.",
            file=sys.stderr,
        )
        return 2

    if config_file and not config_file.is_file():
        print(f"ERROR: Config file not found: {config_file}", file=sys.stderr)
        return 2
    if checkpoint_file and not checkpoint_file.is_file():
        print(f"ERROR: Checkpoint file not found: {checkpoint_file}", file=sys.stderr)
        return 2

    python = sys.executable
    infer_cmd = [
        python,
        str(SCRIPTS_DIR / "infer_cityscapes_local.py"),
        "--input-dir",
        display_path(input_dir),
        "--out-dir",
        display_path(out_dir),
        "--model-info",
        display_path(model_info),
        "--device",
        args.device,
    ]
    if config_file and checkpoint_file:
        infer_cmd.extend(["--config-file", display_path(config_file)])
        infer_cmd.extend(["--checkpoint-file", display_path(checkpoint_file)])

    remap_cmd = [
        python,
        str(SCRIPTS_DIR / "remap_to_project_classes.py"),
        "--input-dir",
        display_path(out_dir / "cityscapes_pred"),
        "--output-dir",
        display_path(out_dir / "project_pred"),
        "--mapping-config",
        "configs/class_mapping.yaml",
    ]
    visualize_cmd = [
        python,
        str(SCRIPTS_DIR / "visualize_project_masks.py"),
        "--image-dir",
        display_path(input_dir),
        "--mask-dir",
        display_path(out_dir / "project_pred"),
        "--vis-dir",
        display_path(out_dir / "project_vis"),
        "--overlay-dir",
        display_path(out_dir / "project_overlay"),
        "--mapping-config",
        "configs/class_mapping.yaml",
        "--alpha",
        str(args.alpha),
    ]
    stats_cmd = [
        python,
        str(SCRIPTS_DIR / "make_class_statistics.py"),
        "--input-dir",
        display_path(out_dir / "project_pred"),
        "--reports-dir",
        display_path(out_dir / "reports"),
        "--mapping-config",
        "configs/class_mapping.yaml",
    ]

    print("Pipeline configuration:")
    print(f"  input_dir: {input_dir}")
    print(f"  out_dir: {out_dir}")
    print(f"  model_info: {model_info}")
    if config_file and checkpoint_file:
        print(f"  config_file: {config_file}")
        print(f"  checkpoint_file: {checkpoint_file}")
    print(f"  device: {args.device}")
    print(f"  alpha: {args.alpha}")

    run_step("1/4 Cityscapes local checkpoint inference", infer_cmd)
    run_step("2/4 Remap Cityscapes masks to project classes", remap_cmd)
    run_step("3/4 Generate project visualizations and overlays", visualize_cmd)
    run_step("4/4 Generate class statistics reports", stats_cmd)

    print("\nPipeline complete.")
    print(f"Outputs written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
