"""python -m neuralplsi: print the version and the engine library, building it if needed."""
from . import __version__
from .native import library, library_path

library()
print(f'neuralplsi {__version__}: {library_path()}')
