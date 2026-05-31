"""Batch inference with local MMSegmentation config and checkpoint files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

CITYSCAPES_PALETTE = [
    (128, 64, 128),   # road
    (244, 35, 232),   # sidewalk
    (70, 70, 70),     # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170, 30),   # traffic light
    (220, 220, 0),    # traffic sign
    (107, 142, 35),   # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),   # sky
    (220, 20, 60),    # person
    (255, 0, 0),      # rider
    (0, 0, 142),      # car
    (0, 0, 70),       # truck
    (0, 60, 100),     # bus
    (0, 80, 100),     # train
    (0, 0, 230),      # motorcycle
    (119, 11, 32),    # bicycle
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batch Cityscapes inference with a local MMSegmentation config "
            "and a local .pth checkpoint. This script never downloads models "
            "during inference."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="input_images",
        help="Directory containing jpg/jpeg/png input images.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Output root directory.",
    )
    parser.add_argument(
        "--model-info",
        default="checkpoints/active_model_info.json",
        help="JSON file recording local config_file and checkpoint_file.",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Explicit local MMSegmentation config file. Must be used with --checkpoint-file.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Explicit local .pth checkpoint file. Must be used with --config-file.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device, for example cuda:0 or cpu.",
    )
    return parser.parse_args()


def resolve_path(path_text: str | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    return (base / path).resolve()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        print(f"ERROR: Cannot find {description}: {path}", file=sys.stderr)
        if description == "model info file":
            print(
                "Please run scripts/00_download_pretrained_model.py first "
                "to create checkpoints/active_model_info.json.",
                file=sys.stderr,
            )
        return_code = 2
        raise SystemExit(return_code)


def load_model_files(args: argparse.Namespace) -> tuple[Path, Path]:
    config_arg = args.config_file
    checkpoint_arg = args.checkpoint_file

    if bool(config_arg) != bool(checkpoint_arg):
        print(
            "ERROR: Please provide both --config-file and --checkpoint-file, "
            "or provide neither and use --model-info.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if config_arg and checkpoint_arg:
        config_file = resolve_path(config_arg)
        checkpoint_file = resolve_path(checkpoint_arg)
    else:
        model_info = resolve_path(args.model_info)
        require_file(model_info, "model info file")
        try:
            info = json.loads(model_info.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid JSON in model info file {model_info}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

        missing_keys = [key for key in ("config_file", "checkpoint_file") if key not in info]
        if missing_keys:
            print(
                f"ERROR: Missing keys in {model_info}: {', '.join(missing_keys)}",
                file=sys.stderr,
            )
            raise SystemExit(2)

        config_file = resolve_path(info["config_file"])
        checkpoint_file = resolve_path(info["checkpoint_file"])

    require_file(config_file, "config file")
    require_file(checkpoint_file, "checkpoint file")
    return config_file, checkpoint_file


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        raise SystemExit(2)

    images = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        print(
            f"ERROR: No jpg/jpeg/png images found in input directory: {input_dir}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return images


def import_mmseg_apis():
    try:
        from mmseg.apis import inference_model, init_model
    except ModuleNotFoundError as exc:
        print(
            "ERROR: MMSegmentation or one of its runtime dependencies is missing. "
            "Please activate the OpenMMLab environment and install the required packages. "
            f"Missing module: {exc.name}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    except Exception as exc:
        print(
            "ERROR: Failed to import MMSegmentation inference APIs. "
            "Please check the active OpenMMLab environment. "
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return init_model, inference_model


def choose_device(device: str) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        print("ERROR: PyTorch is not installed in the active environment.", file=sys.stderr)
        raise SystemExit(2) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "WARNING: CUDA is not available. Falling back to CPU; inference will be slow.",
            file=sys.stderr,
        )
        return "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU; inference will be slow.", file=sys.stderr)
    return device


def extract_pred_mask(result):
    pred_sem_seg = getattr(result, "pred_sem_seg", None)
    if pred_sem_seg is None and isinstance(result, (list, tuple)) and result:
        pred_sem_seg = getattr(result[0], "pred_sem_seg", None)
    if pred_sem_seg is None or not hasattr(pred_sem_seg, "data"):
        print("ERROR: Could not find pred_sem_seg.data in MMSegmentation result.", file=sys.stderr)
        raise SystemExit(3)

    mask = pred_sem_seg.data
    while getattr(mask, "ndim", 0) > 2:
        mask = mask.squeeze(0)
    if getattr(mask, "ndim", 0) != 2:
        print(f"ERROR: Prediction mask is not 2D after squeezing: shape={tuple(mask.shape)}", file=sys.stderr)
        raise SystemExit(3)
    return mask.detach().cpu().to(dtype=__import__("torch").uint8).contiguous()


def tensor_to_l_image(mask) -> Image.Image:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    try:
        return Image.fromarray(mask.numpy(), mode="L")
    except Exception:
        mask_bytes = bytes(mask.view(-1).tolist())
        return Image.frombytes("L", (width, height), mask_bytes)


def colorize_cityscapes(mask_image: Image.Image) -> Image.Image:
    palette_values: list[int] = []
    for color in CITYSCAPES_PALETTE:
        palette_values.extend(color)
    palette_values.extend([0, 0, 0] * (256 - len(CITYSCAPES_PALETTE)))

    paletted = mask_image.convert("P")
    paletted.putpalette(palette_values)
    return paletted.convert("RGB")


def save_outputs(image_path: Path, mask, pred_dir: Path, vis_dir: Path) -> None:
    stem = image_path.stem
    mask_image = tensor_to_l_image(mask)
    mask_path = pred_dir / f"{stem}.png"
    mask_image.save(mask_path)

    original = Image.open(image_path).convert("RGB")
    color_mask = colorize_cityscapes(mask_image)
    if original.size != color_mask.size:
        print(
            f"ERROR: Visualization size mismatch for {image_path.name}: "
            f"image={original.size}, mask={color_mask.size}",
            file=sys.stderr,
        )
        raise SystemExit(3)
    overlay = Image.blend(original, color_mask, alpha=0.5)
    overlay.save(vis_dir / f"{stem}.png")


def main() -> int:
    args = parse_args()

    input_dir = resolve_path(args.input_dir)
    out_dir = resolve_path(args.out_dir)
    cityscapes_vis_dir = out_dir / "cityscapes_vis"
    cityscapes_pred_dir = out_dir / "cityscapes_pred"
    cityscapes_vis_dir.mkdir(parents=True, exist_ok=True)
    cityscapes_pred_dir.mkdir(parents=True, exist_ok=True)

    config_file, checkpoint_file = load_model_files(args)
    images = collect_images(input_dir)
    device = choose_device(args.device)
    init_model, inference_model = import_mmseg_apis()

    print("Using local MMSegmentation files:")
    print(f"  config: {config_file}")
    print(f"  checkpoint: {checkpoint_file}")
    print(f"  device: {device}")
    print(f"Found {len(images)} input image(s).")

    model = init_model(str(config_file), str(checkpoint_file), device=device)

    processed = 0
    for image_path in images:
        print(f"[{processed + 1}/{len(images)}] infer {image_path.name}")
        result = inference_model(model, str(image_path))
        mask = extract_pred_mask(result)
        save_outputs(image_path, mask, cityscapes_pred_dir, cityscapes_vis_dir)
        processed += 1

    print("Inference complete.")
    print(f"Processed images: {processed}")
    print(f"Cityscapes index masks: {cityscapes_pred_dir}")
    print(f"Cityscapes visualizations: {cityscapes_vis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
