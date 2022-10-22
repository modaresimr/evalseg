import glob
import importlib
import sys
from os.path import basename, dirname, isfile

from . import common, compress, ct, geometry, io, metrics, ui, ui3d


def reload():
    for module in list(sys.modules.values()):
        if "evalseg." in f"{module}":
            importlib.reload(module)
