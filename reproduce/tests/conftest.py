"""The reproduce scripts and tests use this checkout's reproduce/ and python/neuralplsi."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]
