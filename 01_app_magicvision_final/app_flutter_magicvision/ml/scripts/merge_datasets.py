"""Merge filtered YOLO datasets into ml/datasets/dataset_final_magicvision/images/all."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SOURCES = [
    "ml/datasets/processed/coco_filtered",
    "ml/datasets/processed/roboflow",
    "ml/datasets/processed/kaggle",
    "ml/datasets/raw/lima_labeled",
]


def find_label_for_image(source_root: Path, image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        label_parts = parts[:index] + ["labels"] + parts[index + 1 :]
        label_path = Path(*label_parts).with_suffix(".txt")
        if label_path.exists():
            return label_path
    candidates = list(source_root.rglob(f"{image_path.stem}.txt"))
    return candidates[0] if candidates else None


def iter_images(source_root: Path) -> list[Path]:
    if not source_root.exists():
        return []
    return sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def unique_name(prefix: str, image_path: Path, used: set[str]) -> str:
    base = f"{prefix}_{image_path.stem}".replace(" ", "_")
    candidate = f"{base}{image_path.suffix.lower()}"
    counter = 1
    while candidate.lower() in used:
        candidate = f"{base}_{counter:04d}{image_path.suffix.lower()}"
        counter += 1
    used.add(candidate.lower())
    return candidate


def merge_sources(sources: list[Path], output: Path) -> int:
    images_output = output / "images" / "all"
    labels_output = output / "labels" / "all"
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    used_names = {path.name.lower() for path in images_output.iterdir() if path.is_file()}
    total = 0
    for source in sources:
        if not source.exists():
            print(f"[WARN] Fuente no encontrada, se omite: {source}")
            continue
        prefix = source.name
        for image_path in iter_images(source):
            label_path = find_label_for_image(source, image_path)
            if label_path is None:
                print(f"[WARN] Imagen sin label, se omite: {image_path}")
                continue
            output_name = unique_name(prefix, image_path, used_names)
            output_image = images_output / output_name
            output_label = labels_output / f"{Path(output_name).stem}.txt"
            shutil.copy2(image_path, output_image)
            shutil.copy2(label_path, output_label)
            total += 1
        print(f"[OK] Fuente procesada: {source}")
    print(f"Total fusionado en images/all: {total}")
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fusiona datasets YOLO filtrados para MagicVision.")
    parser.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES, help="Carpetas fuente.")
    parser.add_argument("--output", default="ml/datasets/dataset_final_magicvision", help="Dataset MagicVision destino.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merge_sources([Path(item) for item in args.sources], Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
