"""Download Kaggle datasets when the Kaggle API is configured.

No dataset slug is hardcoded because Kaggle mirrors, licenses and availability
change often. Pass datasets after checking license and annotation format:
  python ml/scripts/download_kaggle.py --dataset owner/dataset-slug
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SUGGESTED_SEARCHES = [
    "pothole detection bounding boxes YOLO",
    "stairs object detection bounding boxes",
    "obstacle detection visually impaired dataset",
    "BDD100K detection labels",
]


def has_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists() or (
        bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    )


def print_setup_instructions() -> None:
    print("Kaggle API no esta configurada.")
    print("1. En Kaggle: Account > Create New API Token.")
    print("2. Guarda kaggle.json en ~/.kaggle/kaggle.json.")
    print("3. En Linux/Mac: chmod 600 ~/.kaggle/kaggle.json.")
    print("4. Busca datasets y revisa licencia/anotaciones:")
    for term in SUGGESTED_SEARCHES:
        print(f"   - {term}")
    print("5. Ejecuta: python ml/scripts/download_kaggle.py --dataset owner/slug")
    print("BDD100K suele requerir cuenta/terminos propios; usa la pagina oficial si Kaggle no cubre tu licencia.")


def download_dataset(slug: str, output: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Falta el paquete Kaggle. Ejecuta: pip install kaggle")
        return

    dataset_output = output / slug.replace("/", "_")
    dataset_output.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Descargando Kaggle dataset: {slug}")
    api.dataset_download_files(slug, path=str(dataset_output), unzip=True)
    print(f"[OK] Dataset en {dataset_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga datasets desde Kaggle con credenciales locales.")
    parser.add_argument("--dataset", action="append", default=[], help="Slug Kaggle owner/dataset.")
    parser.add_argument("--output", default="ml/datasets/processed/kaggle", help="Carpeta destino.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not has_kaggle_credentials() or not args.dataset:
        print_setup_instructions()
        return 0

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for slug in args.dataset:
        try:
            download_dataset(slug, output)
        except Exception as error:  # noqa: BLE001
            print(f"[WARN] No se pudo descargar {slug}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
