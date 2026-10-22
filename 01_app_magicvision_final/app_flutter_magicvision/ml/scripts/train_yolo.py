"""Train a lightweight YOLO model for MagicVision."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def runtime_data_yaml(data_path: Path) -> Path:
    raw = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_path = Path(raw.get("path", ""))
    if not dataset_path.is_absolute():
        dataset_path = (data_path.parent / dataset_path).resolve()
    raw["path"] = str(dataset_path)
    output = data_path.parent / "data.runtime.yaml"
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLO para deteccion urbana MagicVision.")
    parser.add_argument("--model", default="yolo11n.pt", help="Modelo base Ultralytics.")
    parser.add_argument("--data", default="ml/configs/data.yaml", help="Ruta data.yaml.")
    parser.add_argument("--epochs", type=int, default=80, help="Numero de epocas.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de imagen.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--project", default="runs/detect", help="Carpeta de resultados.")
    parser.add_argument("--name", default="magicvision_yolo", help="Nombre del experimento.")
    parser.add_argument("--device", default=None, help="Dispositivo Ultralytics: 0, cpu, etc.")
    parser.add_argument("--workers", type=int, default=0, help="Workers dataloader; 0 es mas estable en Windows.")
    parser.add_argument("--patience", type=int, default=20, help="Paciencia early stopping.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.data).exists():
        print(f"[WARN] No existe data.yaml: {args.data}")
        print("Prepara el dataset antes de entrenar.")
        return 0

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("Faltan dependencias. Ejecuta: pip install ultralytics torch")
        return 0

    if not torch.cuda.is_available():
        print("[INFO] No se detecto GPU CUDA. Para 80 epocas se recomienda Google Colab con GPU.")

    data_yaml = runtime_data_yaml(Path(args.data).resolve())
    project = Path(args.project).resolve()

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project),
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
    )
    print(f"Entrenamiento finalizado. Revisa {project / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
