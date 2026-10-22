"""Download/filter COCO 2017 classes used by MagicVision.

This script avoids requiring the full COCO download by default. It downloads
annotations when needed and filters images only if the corresponding image files
already exist locally. Use --download-images when running on a machine/Colab
with enough space.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


COCO_URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}

COCO_TO_MAGICVISION = {
    1: 0,  # person
    3: 1,  # car
    4: 2,  # motorcycle
    6: 3,  # bus
    2: 4,  # bicycle
    14: 7,  # parking meter as pole-like vertical street fixture
    10: 8,  # traffic light as traffic_sign/context signal
    13: 8,  # stop sign
    11: 9,  # fire hydrant as sidewalk obstacle
    15: 9,  # bench as sidewalk obstacle
}

SPLIT_TO_OUTPUT = {
    "train2017": "train",
    "val2017": "val",
}


def download_file(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[OK] Ya existe: {destination}")
        return True
    try:
        print(f"Descargando {url}")
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as error:  # noqa: BLE001
        print(f"[WARN] No se pudo descargar {url}: {error}")
        return False


def unzip(zip_path: Path, destination: Path) -> None:
    if not zip_path.exists():
        return
    print(f"Extrayendo {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def ensure_coco(root: Path, download_images: bool, splits: list[str]) -> None:
    annotations_file = root / "annotations" / "instances_train2017.json"
    if not annotations_file.exists():
        zip_path = root / "annotations_trainval2017.zip"
        if download_file(COCO_URLS["annotations"], zip_path):
            unzip(zip_path, root)

    if not download_images:
        return
    for split in splits:
        if split not in ("train2017", "val2017"):
            continue
        split_dir = root / split
        if split_dir.exists() and any(split_dir.iterdir()):
            continue
        zip_path = root / f"{split}.zip"
        if download_file(COCO_URLS[split], zip_path):
            unzip(zip_path, root)


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, box_width, box_height = bbox
    x_center = (x + box_width / 2) / width
    y_center = (y + box_height / 2) / height
    return x_center, y_center, box_width / width, box_height / height


def filter_split(coco_root: Path, output_root: Path, split: str, max_images: int | None) -> int:
    annotation_path = coco_root / "annotations" / f"instances_{split}.json"
    images_root = coco_root / split
    if not annotation_path.exists():
        print(f"[WARN] Falta annotation JSON: {annotation_path}")
        return 0
    if not images_root.exists():
        print(f"[WARN] Faltan imagenes COCO en {images_root}. Descarga {split}.zip o usa --download-images.")
        return 0

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images_by_id = {image["id"]: image for image in data["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        class_id = COCO_TO_MAGICVISION.get(ann.get("category_id"))
        if class_id is None:
            continue
        if ann.get("iscrowd", 0):
            continue
        annotations_by_image[ann["image_id"]].append(ann)

    output_split = SPLIT_TO_OUTPUT.get(split, split)
    output_images = output_root / "images" / output_split
    output_labels = output_root / "labels" / output_split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    saved = 0
    for image_id, annotations in annotations_by_image.items():
        image = images_by_id.get(image_id)
        if not image:
            continue
        source_image = images_root / image["file_name"]
        if not source_image.exists():
            continue
        stem = source_image.stem
        label_lines = []
        for ann in annotations:
            class_id = COCO_TO_MAGICVISION[ann["category_id"]]
            x_center, y_center, box_width, box_height = coco_bbox_to_yolo(
                ann["bbox"],
                image["width"],
                image["height"],
            )
            if box_width <= 0 or box_height <= 0:
                continue
            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )
        if not label_lines:
            continue
        shutil.copy2(source_image, output_images / source_image.name)
        (output_labels / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        saved += 1
        if max_images is not None and saved >= max_images:
            break
    print(f"[OK] {split}: {saved} imagenes filtradas")
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga/filtra COCO 2017 a formato YOLO.")
    parser.add_argument("--coco-root", default="ml/datasets/raw/coco", help="Carpeta COCO local.")
    parser.add_argument("--output", default="ml/datasets/processed/coco_filtered", help="Salida filtrada.")
    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"], help="Splits COCO.")
    parser.add_argument("--download-images", action="store_true", help="Descarga train2017/val2017 completos.")
    parser.add_argument("--max-images", type=int, default=None, help="Limite por split para pruebas.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coco_root = Path(args.coco_root)
    output_root = Path(args.output)
    ensure_coco(coco_root, args.download_images, args.splits)
    total = 0
    for split in args.splits:
        total += filter_split(coco_root, output_root, split, args.max_images)
    print(f"Total COCO filtrado: {total}")
    if total == 0:
        print("No se copiaron imagenes. En Colab usa --download-images o coloca COCO en ml/datasets/raw/coco.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
