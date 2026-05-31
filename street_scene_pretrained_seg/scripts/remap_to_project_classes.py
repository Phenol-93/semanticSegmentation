"""Remap Cityscapes index masks to the project's 9-class index masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASK_SUFFIXES = {".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remap single-channel Cityscapes prediction masks into project "
            "class masks. This is a post-processing step only; it does not "
            "train or modify the original Cityscapes predictions."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/cityscapes_pred",
        help="Directory containing Cityscapes 19-class index masks.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/project_pred",
        help="Directory for remapped project-class index masks.",
    )
    parser.add_argument(
        "--mapping-config",
        default="configs/class_mapping.yaml",
        help="YAML file defining Cityscapes-to-project class mapping.",
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


def load_mapping(config_path: Path) -> tuple[dict[int, int], dict[int, str], int]:
    if not config_path.is_file():
        print(f"ERROR: Mapping config not found: {config_path}", file=sys.stderr)
        raise SystemExit(2)

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"ERROR: Invalid YAML in mapping config {config_path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    raw_mapping = data.get("cityscapes_to_project")
    if not isinstance(raw_mapping, dict):
        print("ERROR: class_mapping.yaml must contain cityscapes_to_project mapping.", file=sys.stderr)
        raise SystemExit(2)

    mapping = {int(city_id): int(project_id) for city_id, project_id in raw_mapping.items()}
    project_names = {
        int(item["id"]): str(item["name"])
        for item in data.get("project_classes", [])
        if "id" in item and "name" in item
    }
    ignore_index = int(data.get("ignore_index", 255))

    expected_project_ids = set(range(9)) | {ignore_index}
    invalid_targets = sorted(set(mapping.values()) - expected_project_ids)
    if invalid_targets:
        print(
            f"ERROR: Mapping contains invalid project class IDs: {invalid_targets}. "
            f"Allowed IDs are 0-8 and {ignore_index}.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if 2 not in project_names:
        print("ERROR: project_classes must include lane_marking with id 2.", file=sys.stderr)
        raise SystemExit(2)

    return mapping, project_names, ignore_index


def collect_masks(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        raise SystemExit(2)

    masks = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in MASK_SUFFIXES
    ]
    if not masks:
        print(f"ERROR: No PNG masks found in input directory: {input_dir}", file=sys.stderr)
        raise SystemExit(2)
    return masks


def read_index_mask(mask_path: Path) -> np.ndarray:
    image = Image.open(mask_path)
    array = np.array(image)
    if array.ndim != 2:
        print(
            f"ERROR: Input mask must be a single-channel class-index image: "
            f"{mask_path} has shape {array.shape} and mode {image.mode}.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not np.issubdtype(array.dtype, np.integer):
        print(
            f"ERROR: Input mask must contain integer class IDs: {mask_path} dtype={array.dtype}.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return array


def remap_mask(mask: np.ndarray, mapping: dict[int, int], ignore_index: int, mask_path: Path) -> np.ndarray:
    values = set(int(value) for value in np.unique(mask))
    allowed_input_values = set(mapping.keys()) | {ignore_index}
    invalid_values = sorted(values - allowed_input_values)
    if invalid_values:
        print(
            f"ERROR: {mask_path.name} contains values not present in Cityscapes mapping: "
            f"{invalid_values}. Expected Cityscapes IDs 0-18.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    output = np.full(mask.shape, ignore_index, dtype=np.uint8)
    for city_id, project_id in mapping.items():
        output[mask == city_id] = project_id
    output[mask == ignore_index] = ignore_index

    output_values = set(int(value) for value in np.unique(output))
    allowed_output_values = set(range(9)) | {ignore_index}
    invalid_outputs = sorted(output_values - allowed_output_values)
    if invalid_outputs:
        print(
            f"ERROR: Remapped mask for {mask_path.name} contains invalid project IDs: "
            f"{invalid_outputs}.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    return output


def format_present_classes(mask: np.ndarray, project_names: dict[int, str]) -> str:
    values = [int(value) for value in np.unique(mask)]
    parts = []
    for value in values:
        name = project_names.get(value, f"unknown_{value}")
        parts.append(f"{value}:{name}")
    return ", ".join(parts)


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    mapping_config = resolve_path(args.mapping_config)

    mapping, project_names, ignore_index = load_mapping(mapping_config)
    masks = collect_masks(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mapping config: {mapping_config}")
    print(f"Input Cityscapes masks: {input_dir}")
    print(f"Output project masks: {output_dir}")
    print(
        "Note: lane_marking (id 2) is kept in the project schema, but Cityscapes "
        "pretrained models do not produce it as a stable source class."
    )

    for index, mask_path in enumerate(masks, start=1):
        mask = read_index_mask(mask_path)
        remapped = remap_mask(mask, mapping, ignore_index, mask_path)
        output_path = output_dir / mask_path.name
        Image.fromarray(remapped, mode="L").save(output_path)
        print(
            f"[{index}/{len(masks)}] {mask_path.name} -> {output_path.name}; "
            f"project classes: {format_present_classes(remapped, project_names)}"
        )

    print(f"Remapping complete. Processed masks: {len(masks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
