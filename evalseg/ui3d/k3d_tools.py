from itertools import product
from k3d.platonic import PlatonicSolid
import k3d
import numpy as np
from k3d.colormaps import basic_color_maps, matplotlib_color_maps
from tqdm.auto import tqdm

from .. import common, geometry


def colormap_convert(arr):
    arr = np.multiply(arr, 255).astype(int)
    tall = [
        (arr[x] << 24) + (arr[x + 1] << 16) + (arr[x + 2] << 8) + (arr[x + 3])
        for x in range(0, len(arr), 4)
    ]
    # if indx==None:
    return common.CircleList(tall)


# colormap=cm(basic_color_maps.RainbowDesaturated)
cls_colormap = colormap_convert(matplotlib_color_maps.tab10)[::28]
tp_colormap = colormap_convert(matplotlib_color_maps.Greens)[28 * 3:: 28]
fn_colormap = colormap_convert(matplotlib_color_maps.Blues)[28 * 3:: 28]
fp2_colormap = colormap_convert(matplotlib_color_maps.spring)[28 * 3:: 28]
fp_colormap = colormap_convert(matplotlib_color_maps.Reds)[28 * 3:: 28]

bone_colormap = colormap_convert(matplotlib_color_maps.bone)


class Cube(k3d.platonic.Cube):
    """Create a cube.
    """

    def __init__(self, origin=[0, 0, 0], size=[1, 1, 1]):
        """Inits Cube with an origin and a size.

        Args:
            origin (list, optional): The position of centroid of the solid. Defaults to [0, 0, 0].
            size (int array, optional): The size*sqrt(3) is the distance of each vertex from centroid of the solid. Defaults to 1.

        Raises:
            TypeError: Origin attribute should have 3 coordinates.
        """

        super().__init__([1, 1, 1], 1)
        from itertools import product

        for i in range(3):
            self.vertices[:, i] = self.vertices[:, i]*size[i]/2+origin[i]
