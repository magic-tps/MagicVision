"""Extract frames from local videos for the MagicVision custom dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def iter_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    if not path.exists():
        return []
    return sorted(file for file in path.rglob("*") if file.suffix.lower() in VIDEO_EXTENSIONS)


def extract_frames(input_path: Path, output_path: Path, every_seconds: float, max_frames: int | None) -> int:
    try:
        import cv2
    except ImportError:
        print("OpenCV no esta instalado. Ejecuta: pip install opencv-python")
        return 0

    output_path.mkdir(parents=True, exist_ok=True)
    total = 0
    for video_path in iter_videos(input_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] No pude abrir video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(round(fps * every_seconds)))
        frame_index = 0
        saved_for_video = 0
        video_stem = video_path.stem.replace(" ", "_")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                output_file = output_path / f"{video_stem}_{frame_index:08d}.jpg"
                cv2.imwrite(str(output_file), frame)
                total += 1
                saved_for_video += 1
                if max_frames is not None and total >= max_frames:
                    cap.release()
                    print(f"[OK] {video_path}: {saved_for_video} frames")
                    return total
            frame_index += 1

        cap.release()
        print(f"[OK] {video_path}: {saved_for_video} frames")
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae frames de videos propios.")
    parser.add_argument("--input", default="ml/datasets/raw/videos", help="Carpeta o archivo de video.")
    parser.add_argument("--output", default="ml/datasets/raw/frames", help="Carpeta destino.")
    parser.add_argument("--every_seconds", type=float, default=2.0, help="Frecuencia de extraccion.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximo total de frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total = extract_frames(Path(args.input), Path(args.output), args.every_seconds, args.max_frames)
    print(f"Frames guardados: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
