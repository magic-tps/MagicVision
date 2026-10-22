"""Convert the Figshare pothole dataset to MagicVision YOLO layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POTHOLE_CLASS_ID = 6


def convert_label(source: Path, destination: Path) -> bool:
    output_lines = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        output_lines.append(" ".join([str(POTHOLE_CLASS_ID), *parts[1:]]))
    if not output_lines:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return True


def prepare(input_root: Path, output_root: Path) -> int:
    images_output = output_root / "images" / "all"
    labels_output = output_root / "labels" / "all"
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    total = 0
    for image_path in sorted(input_root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        safe_stem = "_".join(image_path.relative_to(input_root).with_suffix("").parts)
        output_image = images_output / f"figshare_{safe_stem}{image_path.suffix.lower()}"
        output_label = labels_output / f"figshare_{safe_stem}.txt"
        if not convert_label(label_path, output_label):
            continue
        shutil.copy2(image_path, output_image)
        total += 1
    print(f"[OK] Figshare potholes convertidos: {total}")
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convierte potholes Figshare a MagicVision YOLO.")
    parser.add_argument(
        "--input",
        default="ml/datasets/raw/figshare/extracted/Potholes_dataset",
        help="Carpeta extraida Potholes_dataset.",
    )
    parser.add_argument(
        "--output",
        default="ml/datasets/processed/figshare_potholes",
        help="Salida YOLO MagicVision.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare(Path(args.input), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
