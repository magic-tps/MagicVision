"""Filter/remap YOLO labels to the MagicVision 10-class taxonomy."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TARGET_CLASSES = [
    "person",
    "car",
    "motorcycle",
    "bus",
    "bicycle",
    "stairs",
    "pothole",
    "pole",
    "traffic_sign",
    "obstacle",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ALIASES = {
    "persons": "person",
    "people": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "cars": "car",
    "vehicle": "car",
    "vehicles": "car",
    "taxi": "car",
    "taxis": "car",
    "buses": "bus",
    "motorcycles": "motorcycle",
    "motorbikes": "motorcycle",
    "bicycles": "bicycle",
    "bike": "bicycle",
    "bikes": "bicycle",
    "stair": "stairs",
    "stairs": "stairs",
    "steps": "stairs",
    "hole": "pothole",
    "holes": "pothole",
    "potholes": "pothole",
    "electric_pole": "pole",
    "poles": "pole",
    "traffic_sign": "traffic_sign",
    "traffic_signs": "traffic_sign",
    "sign": "traffic_sign",
    "signs": "traffic_sign",
    "fence": "obstacle",
    "fences": "obstacle",
    "bump": "obstacle",
    "bumps": "obstacle",
    "barrier": "obstacle",
    "barriers": "obstacle",
    "roadblock": "obstacle",
    "roadblocks": "obstacle",
    "obstacles": "obstacle",
}


def read_classes(path: Path) -> list[str]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = data.get("names", [])
        if isinstance(names, dict):
            return [names[index] for index in sorted(names)]
        return [str(name) for name in names]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def build_mapping(
    source_classes: list[str],
    target_classes: list[str],
    explicit_maps: list[str],
) -> dict[int, int]:
    target_by_name = {name: index for index, name in enumerate(target_classes)}
    mapping: dict[int, int] = {}
    aliases = dict(DEFAULT_ALIASES)

    for item in explicit_maps:
        if "=" not in item:
            raise ValueError(f"Mapa invalido '{item}'. Usa source=target.")
        source, target = item.split("=", 1)
        aliases[normalize(source)] = normalize(target)

    for index, name in enumerate(source_classes):
        normalized = normalize(name)
        target_name = aliases.get(normalized, normalized)
        if target_name in target_by_name:
            mapping[index] = target_by_name[target_name]
    return mapping


def filter_label(source_label: Path, destination_label: Path, mapping: dict[int, int]) -> bool:
    output_lines = []
    for line in source_label.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            source_class = int(float(parts[0]))
        except ValueError:
            continue
        target_class = mapping.get(source_class)
        if target_class is None:
            continue
        output_lines.append(" ".join([str(target_class), *parts[1:]]))
    if not output_lines:
        return False
    destination_label.parent.mkdir(parents=True, exist_ok=True)
    destination_label.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return True


def find_image(images_root: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_root / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    matches = [path for path in images_root.rglob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    return matches[0] if matches else None


def split_roots(input_root: Path, split: str) -> tuple[Path, Path] | None:
    candidates = [
        (input_root / "images" / split, input_root / "labels" / split),
        (input_root / split / "images", input_root / split / "labels"),
    ]
    if split == "val":
        candidates.extend(
            [
                (input_root / "images" / "valid", input_root / "labels" / "valid"),
                (input_root / "valid" / "images", input_root / "valid" / "labels"),
                (input_root / "images" / "validation", input_root / "labels" / "validation"),
                (input_root / "validation" / "images", input_root / "validation" / "labels"),
            ]
        )
    for images_root, labels_root in candidates:
        if labels_root.exists() and images_root.exists():
            return images_root, labels_root
    return None


def filter_dataset(
    input_root: Path,
    output_root: Path,
    source_classes: Path,
    target_classes: Path,
    explicit_maps: list[str],
) -> int:
    source = read_classes(source_classes)
    target = read_classes(target_classes) if target_classes.exists() else TARGET_CLASSES
    mapping = build_mapping(source, target, explicit_maps)
    if not mapping:
        print("[WARN] No hay clases comunes entre source_classes y target_classes.")
        return 0

    total = 0
    for split in ("train", "val", "test"):
        roots = split_roots(input_root, split)
        if roots is None:
            continue
        images_root, labels_root = roots
        for label_path in labels_root.rglob("*.txt"):
            relative = label_path.relative_to(labels_root)
            destination_label = output_root / "labels" / split / relative
            if not filter_label(label_path, destination_label, mapping):
                continue
            image_path = find_image(images_root, label_path.stem)
            if image_path is None:
                print(f"[WARN] Label filtrado sin imagen: {label_path}")
                continue
            destination_image = output_root / "images" / split / image_path.name
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination_image)
            total += 1
    print(f"Imagenes filtradas: {total}")
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filtra clases YOLO hacia la taxonomia MagicVision.")
    parser.add_argument("--input", required=True, help="Dataset YOLO origen.")
    parser.add_argument("--output", required=True, help="Dataset YOLO destino.")
    parser.add_argument("--source-classes", required=True, help="Archivo de clases del origen.")
    parser.add_argument("--target-classes", default="ml/configs/classes.txt", help="Archivo de clases MagicVision.")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Remapeo extra source=target. Ejemplo: vehicles=car",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    filter_dataset(
        Path(args.input),
        Path(args.output),
        Path(args.source_classes),
        Path(args.target_classes),
        args.map,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
