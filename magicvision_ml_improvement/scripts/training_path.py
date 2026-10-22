"""Compatibility shim for scripts that need the sibling training directory."""
from pathlib import Path
import sys


def ensure_imports():
    training = str(Path(__file__).resolve().parents[1] / 'training')
    if training not in sys.path:
        sys.path.insert(0, training)


ensure_imports()
