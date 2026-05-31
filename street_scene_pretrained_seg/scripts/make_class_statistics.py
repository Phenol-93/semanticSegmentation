"""Create class-ratio statistics from project prediction masks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pixel counts and ratios from project-class prediction masks. "
            "This script does not compute mIoU because no ground-truth masks are used."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/project_pred",
        help="Directory containing project-class index masks.",
    )
    parser.add_argument(
        "--reports-dir",
        default="outputs/reports",
        help="Directory for class_statistics.csv and class_statistics_summary.md.",
    )
    parser.add_argument(
        "--mapping-config",
        default="configs/class_mapping.yaml",
        help="YAML file containing project class definitions.",
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


def load_project_classes(config_path: Path) -> tuple[list[dict], int]:
    if not config_path.is_file():
        print(f"ERROR: Mapping config not found: {config_path}", file=sys.stderr)
        raise SystemExit(2)

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"ERROR: Invalid YAML in mapping config {config_path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    ignore_index = int(data.get("ignore_index", 255))
    classes = [
        {"id": int(item["id"]), "name": str(item["name"])}
        for item in data.get("project_classes", [])
        if int(item.get("id", -1)) != ignore_index
    ]
    classes.sort(key=lambda item: item["id"])

    expected_ids = list(range(9))
    found_ids = [item["id"] for item in classes]
    if found_ids != expected_ids:
        print(
            f"ERROR: Project class IDs must be 0-8 in order. Found: {found_ids}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return classes, ignore_index


def collect_masks(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        raise SystemExit(2)
    masks = sorted(path for path in input_dir.glob("*.png") if path.is_file())
    if not masks:
        print(f"ERROR: No project mask PNG files found in: {input_dir}", file=sys.stderr)
        raise SystemExit(2)
    return masks


def read_mask(mask_path: Path, ignore_index: int) -> np.ndarray:
    image = Image.open(mask_path)
    array = np.array(image)
    if array.ndim != 2:
        print(
            f"ERROR: Project mask must be single-channel: {mask_path} "
            f"has shape {array.shape} and mode {image.mode}.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    values = set(int(value) for value in np.unique(array))
    allowed = set(range(9)) | {ignore_index}
    invalid = sorted(values - allowed)
    if invalid:
        print(
            f"ERROR: Project mask {mask_path.name} contains invalid values: {invalid}. "
            f"Allowed values are 0-8 and {ignore_index}.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return array


def ratio(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def make_row(mask_path: Path, mask: np.ndarray, classes: list[dict], ignore_index: int) -> dict:
    total_pixels = int(mask.size)
    row = {
        "image_name": mask_path.name,
        "total_pixels": total_pixels,
    }
    for class_item in classes:
        name = class_item["name"]
        class_id = class_item["id"]
        count = int(np.count_nonzero(mask == class_id))
        row[f"{name}_count"] = count
        row[f"{name}_ratio"] = ratio(count, total_pixels)

    ignore_count = int(np.count_nonzero(mask == ignore_index))
    row["ignore_count"] = ignore_count
    row["ignore_ratio"] = ratio(ignore_count, total_pixels)
    return row


def write_csv(csv_path: Path, rows: list[dict], classes: list[dict]) -> None:
    fieldnames = ["image_name", "total_pixels"]
    for class_item in classes:
        name = class_item["name"]
        fieldnames.extend([f"{name}_count", f"{name}_ratio"])
    fieldnames.extend(["ignore_count", "ignore_ratio"])

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = row.copy()
            for key, value in row.items():
                if key.endswith("_ratio"):
                    formatted[key] = f"{value:.6f}"
            writer.writerow(formatted)


def summarize(rows: list[dict], classes: list[dict], ignore_index: int) -> tuple[list[tuple[str, int, float]], list[str]]:
    total_pixels = sum(int(row["total_pixels"]) for row in rows)
    totals: list[tuple[str, int, float]] = []
    for class_item in classes:
        name = class_item["name"]
        count = sum(int(row[f"{name}_count"]) for row in rows)
        totals.append((name, count, ratio(count, total_pixels)))

    ignore_count = sum(int(row["ignore_count"]) for row in rows)
    totals_with_ignore = totals + [("ignore", ignore_count, ratio(ignore_count, total_pixels))]

    near_zero = [name for name, count, _ in totals if count == 0]
    if not near_zero:
        near_zero = [name for name, _, class_ratio in totals if class_ratio < 0.001]
    return totals_with_ignore, near_zero


def write_summary(summary_path: Path, rows: list[dict], classes: list[dict], ignore_index: int) -> None:
    totals, near_zero = summarize(rows, classes, ignore_index)
    class_totals = [item for item in totals if item[0] != "ignore"]
    ranked = sorted(class_totals, key=lambda item: item[1], reverse=True)
    most_common = ranked[:3]
    lane_total = next((count for name, count, _ in class_totals if name == "lane_marking"), 0)

    lines = [
        "# Class Statistics Summary",
        "",
        f"Processed images: {len(rows)}",
        "",
        "This report summarizes predicted project-class masks only. It does not compute mIoU, Pixel Accuracy, or any ground-truth-based metric because this project stage does not use manually annotated masks.",
        "",
        "## Most Common Classes",
        "",
    ]
    for name, count, class_ratio in most_common:
        lines.append(f"- {name}: {count} pixels ({class_ratio:.2%})")

    lines.extend(["", "## Classes With Little Or No Presence", ""])
    if near_zero:
        for name in near_zero:
            lines.append(f"- {name}")
    else:
        lines.append("- None among the 9 project classes.")

    if lane_total == 0:
        lines.extend(
            [
                "",
                "## lane_marking Note",
                "",
                "`lane_marking` is 0 in all processed masks. This is an expected limitation of Cityscapes pretrained semantic segmentation models: they normally do not output lane markings as an independent class, so lane markings are usually predicted as `road`.",
            ]
        )

    lines.extend(["", "## Overall Pixel Ratios", ""])
    for name, count, class_ratio in totals:
        lines.append(f"- {name}: {count} pixels ({class_ratio:.2%})")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    reports_dir = resolve_path(args.reports_dir)
    mapping_config = resolve_path(args.mapping_config)

    classes, ignore_index = load_project_classes(mapping_config)
    masks = collect_masks(input_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mask_path in masks:
        mask = read_mask(mask_path, ignore_index)
        rows.append(make_row(mask_path, mask, classes, ignore_index))

    csv_path = reports_dir / "class_statistics.csv"
    summary_path = reports_dir / "class_statistics_summary.md"
    write_csv(csv_path, rows, classes)
    write_summary(summary_path, rows, classes, ignore_index)

    print(f"Processed masks: {len(rows)}")
    print(f"CSV report: {csv_path}")
    print(f"Markdown summary: {summary_path}")
    print("No mIoU was computed because no ground-truth masks are used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
