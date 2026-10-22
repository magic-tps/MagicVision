"""Validate a YOLO dataset for MagicVision.

Expected structure:
  dataset/
    images/{train,val,test}
    labels/{train,val,test}
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_classes(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_key(path: Path) -> str:
    return path.stem.lower()


def collect_images(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    if not root.exists():
        return images
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        key = image_key(path)
        if key in images:
            duplicates[key].append(path)
        else:
            images[key] = path
    for key, paths in duplicates.items():
        paths.insert(0, images[key])
        print(f"[WARN] Nombre de imagen duplicado '{key}':")
        for item in paths:
            print(f"       - {item}")
    return images


def collect_labels(root: Path) -> dict[str, Path]:
    labels: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    if not root.exists():
        return labels
    for path in root.rglob("*.txt"):
        if not path.is_file():
            continue
        key = image_key(path)
        if key in labels:
            duplicates[key].append(path)
        else:
            labels[key] = path
    for key, paths in duplicates.items():
        paths.insert(0, labels[key])
        print(f"[WARN] Nombre de label duplicado '{key}':")
        for item in paths:
            print(f"       - {item}")
    return labels


def validate_label_file(path: Path, class_count: int) -> tuple[int, int]:
    errors = 0
    warnings = 0
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        print(f"[WARN] Label vacio: {path}")
        return errors, warnings + 1

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            print(f"[ERROR] Label invalido {path}:{line_number}. Se esperan 5 columnas YOLO.")
            errors += 1
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            print(f"[ERROR] Label con valores no numericos {path}:{line_number}.")
            errors += 1
            continue

        if class_id < 0 or (class_count and class_id >= class_count):
            print(f"[ERROR] Clase fuera de rango {path}:{line_number}: {class_id}")
            errors += 1
        if width <= 0 or height <= 0 or width > 1 or height > 1:
            print(f"[ERROR] Caja con ancho/alto invalido {path}:{line_number}: {width}, {height}")
            errors += 1
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            print(f"[ERROR] Centro fuera de rango {path}:{line_number}: {x_center}, {y_center}")
            errors += 1
        if x_center - width / 2 < -0.02 or x_center + width / 2 > 1.02:
            print(f"[ERROR] Caja excede horizontalmente {path}:{line_number}.")
            errors += 1
        if y_center - height / 2 < -0.02 or y_center + height / 2 > 1.02:
            print(f"[ERROR] Caja excede verticalmente {path}:{line_number}.")
            errors += 1
    return errors, warnings


def validate_dataset(dataset: Path, classes_path: Path, splits: list[str]) -> int:
    classes = read_classes(classes_path)
    total_errors = 0
    total_warnings = 0
    all_names: Counter[str] = Counter()

    for split in splits:
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        print(f"\n== Split {split} ==")
        print(f"Imagenes: {images_dir}")
        print(f"Labels:   {labels_dir}")

        images = collect_images(images_dir)
        labels = collect_labels(labels_dir)

        for key, image_path in images.items():
            all_names[key] += 1
            if key not in labels:
                print(f"[WARN] Imagen sin label: {image_path}")
                total_warnings += 1
        for key, label_path in labels.items():
            if key not in images:
                print(f"[ERROR] Label sin imagen: {label_path}")
                total_errors += 1
            errors, warnings = validate_label_file(label_path, len(classes))
            total_errors += errors
            total_warnings += warnings

        unsupported = [
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
        ] if images_dir.exists() else []
        for path in unsupported:
            print(f"[WARN] Extension de imagen no soportada: {path}")
            total_warnings += 1

        print(f"OK split {split}: {len(images)} imagenes, {len(labels)} labels")

    duplicates = [name for name, count in all_names.items() if count > 1]
    for name in duplicates:
        print(f"[WARN] Mismo nombre de imagen aparece en varios splits: {name}")
        total_warnings += 1

    print("\n== Resumen ==")
    print(f"Clases: {len(classes)}")
    print(f"Errores: {total_errors}")
    print(f"Advertencias: {total_warnings}")
    return total_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida un dataset YOLO de MagicVision.")
    parser.add_argument("--dataset", default="ml/datasets/dataset_final_magicvision", help="Raiz del dataset YOLO.")
    parser.add_argument("--classes", default="ml/configs/classes.txt", help="Archivo classes.txt.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits a validar.")
    parser.add_argument("--strict", action="store_true", help="Salir con codigo distinto de cero si hay errores.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_dataset(Path(args.dataset), Path(args.classes), args.splits)
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
