"""Create colored project masks and image overlays from project index masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize project-class index masks using the palette from "
            "configs/class_mapping.yaml. This script does not train a model "
            "or require ground-truth masks."
        )
    )
    parser.add_argument(
        "--image-dir",
        default="input_images",
        help="Directory containing original street-scene images.",
    )
    parser.add_argument(
        "--mask-dir",
        default="outputs/project_pred",
        help="Directory containing project-class index masks.",
    )
    parser.add_argument(
        "--vis-dir",
        default="outputs/project_vis",
        help="Directory for colored project-class masks.",
    )
    parser.add_argument(
        "--overlay-dir",
        default="outputs/project_overlay",
        help="Directory for original image + colored mask overlays.",
    )
    parser.add_argument(
        "--mapping-config",
        default="configs/class_mapping.yaml",
        help="YAML file containing project class palette.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay alpha for the colored mask. Must be between 0 and 1.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_palette(config_path: Path) -> dict[int, tuple[int, int, int]]:
    if not config_path.is_file():
        print(f"ERROR: Mapping config not found: {config_path}", file=sys.stderr)
        raise SystemExit(2)

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"ERROR: Invalid YAML in mapping config {config_path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    palette: dict[int, tuple[int, int, int]] = {}
    for item in data.get("project_classes", []):
        if "id" not in item or "palette" not in item:
            continue
        class_id = int(item["id"])
        color = item["palette"]
        if not isinstance(color, list) or len(color) != 3:
            print(f"ERROR: Invalid palette for class id {class_id}: {color}", file=sys.stderr)
            raise SystemExit(2)
        palette[class_id] = tuple(int(channel) for channel in color)

    required_ids = set(range(9)) | {int(data.get("ignore_index", 255))}
    missing_ids = sorted(required_ids - set(palette))
    if missing_ids:
        print(f"ERROR: Missing palette entries for class IDs: {missing_ids}", file=sys.stderr)
        raise SystemExit(2)

    return palette


def collect_masks(mask_dir: Path) -> list[Path]:
    if not mask_dir.is_dir():
        print(f"ERROR: Mask directory does not exist: {mask_dir}", file=sys.stderr)
        raise SystemExit(2)
    masks = sorted(path for path in mask_dir.glob("*.png") if path.is_file())
    if not masks:
        print(f"ERROR: No project mask PNG files found in: {mask_dir}", file=sys.stderr)
        raise SystemExit(2)
    return masks


def find_image_for_mask(mask_path: Path, image_dir: Path) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{mask_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    print(
        f"ERROR: Could not find original image for mask {mask_path.name} in {image_dir}. "
        "Images and masks must share the same filename prefix.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def read_project_mask(mask_path: Path) -> np.ndarray:
    image = Image.open(mask_path)
    array = np.array(image)
    if array.ndim != 2:
        print(
            f"ERROR: Project mask must be a single-channel class-index image: "
            f"{mask_path} has shape {array.shape} and mode {image.mode}.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    values = set(int(value) for value in np.unique(array))
    allowed = set(range(9)) | {255}
    invalid = sorted(values - allowed)
    if invalid:
        print(
            f"ERROR: Project mask {mask_path.name} contains invalid class IDs: {invalid}. "
            "Allowed values are 0-8 and 255.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return array


def colorize_mask(mask: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> Image.Image:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in palette.items():
        color[mask == class_id] = rgb
    return Image.fromarray(color, mode="RGB")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        print("ERROR: --alpha must be between 0 and 1.", file=sys.stderr)
        raise SystemExit(2)

    image_dir = resolve_path(args.image_dir)
    mask_dir = resolve_path(args.mask_dir)
    vis_dir = resolve_path(args.vis_dir)
    overlay_dir = resolve_path(args.overlay_dir)
    mapping_config = resolve_path(args.mapping_config)

    if not image_dir.is_dir():
        print(f"ERROR: Image directory does not exist: {image_dir}", file=sys.stderr)
        raise SystemExit(2)

    palette = load_palette(mapping_config)
    masks = collect_masks(mask_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image directory: {image_dir}")
    print(f"Project mask directory: {mask_dir}")
    print(f"Colored mask output: {vis_dir}")
    print(f"Overlay output: {overlay_dir}")
    print(f"Overlay alpha: {args.alpha}")

    for index, mask_path in enumerate(masks, start=1):
        image_path = find_image_for_mask(mask_path, image_dir)
        original = Image.open(image_path).convert("RGB")
        mask = read_project_mask(mask_path)
        if original.size != (mask.shape[1], mask.shape[0]):
            print(
                f"ERROR: Size mismatch for {mask_path.name}: "
                f"image {image_path.name} size={original.size}, "
                f"mask size={(mask.shape[1], mask.shape[0])}. "
                "Refusing to resize silently.",
                file=sys.stderr,
            )
            raise SystemExit(2)

        color_mask = colorize_mask(mask, palette)
        vis_path = vis_dir / f"{mask_path.stem}.png"
        overlay_path = overlay_dir / f"{mask_path.stem}.png"
        color_mask.save(vis_path)
        Image.blend(original, color_mask, alpha=args.alpha).save(overlay_path)
        print(f"[{index}/{len(masks)}] {image_path.name} -> {vis_path.name}, {overlay_path.name}")

    print(f"Visualization complete. Processed masks: {len(masks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
