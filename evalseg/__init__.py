import glob
import importlib
import sys
from os.path import basename, dirname, isfile

from . import common, compress, ct_helper, geometry, io, metrics, ui, ui3d, cli


def reload():
    for module in list(sys.modules.values()):
        if "evalseg." in f"{module}" and 'segment_array' not in f"{module}":
            importlib.reload(module)
