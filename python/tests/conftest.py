"""Use the source tree when the package is not installed (it then compiles core/ on first use)."""
import importlib.util
from pathlib import Path
import sys

if importlib.util.find_spec('neuralplsi') is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
