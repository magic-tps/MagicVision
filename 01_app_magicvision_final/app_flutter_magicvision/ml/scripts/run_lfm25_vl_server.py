"""Run an OpenAI-compatible LFM2.5-VL-450M server for MagicVision.

The Flutter app reads the endpoint with:

flutter run --dart-define=MAGICVISION_VLM_ENDPOINT=http://<host>:8000

Use this script on a machine with enough RAM/VRAM. It does not invent
credentials and it does not download private assets; public Hugging Face
weights are fetched by vLLM/SGLang when their runtimes are installed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


DEFAULT_MODEL = "LiquidAI/LFM2.5-VL-450M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Levanta LFM2.5-VL-450M para MagicVision.",
    )
    parser.add_argument("--backend", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime comandos, no inicia el servidor.",
    )
    return parser.parse_args()


def command_for(args: argparse.Namespace) -> list[str]:
    if args.backend == "vllm":
        return [
            "vllm",
            "serve",
            args.model,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]

    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]


def main() -> int:
    args = parse_args()
    cmd = command_for(args)
    endpoint = f"http://{args.host if args.host != '0.0.0.0' else '<IP_DEL_SERVIDOR>'}:{args.port}"

    print("Modelo:", args.model)
    print("Endpoint para Flutter:", endpoint)
    print(
        "Flutter:",
        "flutter run "
        f"--dart-define=MAGICVISION_VLM_ENDPOINT={endpoint}",
    )
    print("Servidor:", " ".join(cmd))

    if args.dry_run:
        return 0

    if args.backend == "vllm" and shutil.which("vllm") is None:
        print("Falta vLLM. Instala con: pip install vllm", file=sys.stderr)
        return 0

    if args.backend == "sglang":
        try:
            __import__("sglang")
        except ImportError:
            print("Falta SGLang. Instala con: pip install sglang", file=sys.stderr)
            return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
