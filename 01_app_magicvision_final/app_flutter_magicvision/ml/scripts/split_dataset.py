"""Split MagicVision YOLO data into train/val/test."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_root: Path) -> list[Path]:
    if not images_root.exists():
        return []
    return sorted(path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_pair(image: Path, label: Path, dataset: Path, split: str) -> None:
    image_output = dataset / "images" / split / image.name
    label_output = dataset / "labels" / split / f"{image.stem}.txt"
    image_output.parent.mkdir(parents=True, exist_ok=True)
    label_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_output)
    shutil.copy2(label, label_output)


def split_dataset(dataset: Path, train: float, val: float, test: float, seed: int) -> int:
    total_ratio = train + val + test
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("--train + --val + --test debe sumar 1.0")

    source_images = list_images(dataset / "images" / "all")
    source_labels = dataset / "labels" / "all"
    if not source_images:
        print("[WARN] No hay imagenes en images/all. Ejecuta merge_datasets.py primero.")
        return 0

    pairs: list[tuple[Path, Path]] = []
    for image in source_images:
        label = source_labels / f"{image.stem}.txt"
        if not label.exists():
            print(f"[WARN] Imagen sin label, se omite: {image}")
            continue
        pairs.append((image, label))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    train_end = int(len(pairs) * train)
    val_end = train_end + int(len(pairs) * val)
    buckets = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }
    for split, split_pairs in buckets.items():
        for image, label in split_pairs:
            copy_pair(image, label, dataset, split)
        print(f"[OK] {split}: {len(split_pairs)} imagenes")
    return len(pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Divide dataset YOLO en train/val/test.")
    parser.add_argument("--dataset", default="ml/datasets/dataset_final_magicvision", help="Raiz del dataset.")
    parser.add_argument("--train", type=float, default=0.7, help="Proporcion train.")
    parser.add_argument("--val", type=float, default=0.2, help="Proporcion val.")
    parser.add_argument("--test", type=float, default=0.1, help="Proporcion test.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_dataset(Path(args.dataset), args.train, args.val, args.test, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
