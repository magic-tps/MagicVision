"""Download Roboflow Universe datasets when credentials and project slugs exist.

The script does not invent dataset IDs. Pass real projects from Roboflow Universe:
  python ml/scripts/download_roboflow.py --dataset workspace/project/version
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SUGGESTED_SEARCHES = [
    "pothole detection",
    "stairs detection",
    "obstacle detection visually impaired",
    "traffic sign detection",
    "pole detection street",
]


def parse_dataset(value: str) -> tuple[str, str, int]:
    parts = value.strip().split("/")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Usa formato workspace/project/version")
    workspace, project, version_text = parts
    try:
        version = int(version_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("version debe ser un entero") from error
    return workspace, project, version


def print_manual_instructions() -> None:
    print("ROBOFLOW_API_KEY no esta configurada o no se indicaron datasets reales.")
    print("1. Entra a https://universe.roboflow.com/")
    print("2. Busca datasets con estos terminos:")
    for term in SUGGESTED_SEARCHES:
        print(f"   - {term}")
    print("3. Revisa licencia, clases y calidad de labels.")
    print("4. Copia workspace/project/version y ejecuta:")
    print("   set ROBOFLOW_API_KEY=tu_key")
    print("   python ml/scripts/download_roboflow.py --dataset workspace/project/1")
    print("5. Descarga siempre en formato YOLOv8 cuando lo hagas manualmente.")


def download_dataset(api_key: str, dataset: tuple[str, str, int], output: Path, fmt: str) -> None:
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Falta el SDK de Roboflow. Ejecuta: pip install roboflow")
        return

    workspace, project, version = dataset
    location = output / f"{workspace}_{project}_v{version}"
    location.mkdir(parents=True, exist_ok=True)
    print(f"Descargando Roboflow: {workspace}/{project}/{version}")
    rf = Roboflow(api_key=api_key)
    rf.workspace(workspace).project(project).version(version).download(fmt, location=str(location))
    print(f"[OK] Dataset en {location}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga datasets reales de Roboflow en YOLOv8.")
    parser.add_argument("--dataset", action="append", type=parse_dataset, default=[], help="workspace/project/version")
    parser.add_argument("--output", default="ml/datasets/processed/roboflow", help="Carpeta destino.")
    parser.add_argument("--format", default="yolov8", help="Formato Roboflow, normalmente yolov8.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key or not args.dataset:
        print_manual_instructions()
        return 0

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for dataset in args.dataset:
        try:
            download_dataset(api_key, dataset, output, args.format)
        except Exception as error:  # noqa: BLE001
            print(f"[WARN] No se pudo descargar {dataset}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
