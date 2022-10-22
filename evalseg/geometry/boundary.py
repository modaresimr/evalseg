import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)

from ..common import Cache
from .roi import one_roi


@Cache.memoize()
def find_binary_boundary(binary_img, mode="thick"):
    """Return bool array where boundaries between labeled regions are True.
    Parameters
    ----------
    binary_img : array of int or bool
        An array in which different regions are labeled with either different
        integers or boolean values.
    connectivity : int in {1, ..., `label_img.ndim`}, optional
        A pixel is considered a boundary pixel if any of its neighbors
        has a different label. `connectivity` controls which pixels are
        considered neighbors. A connectivity of 1 (default) means
        pixels sharing an edge (in 2D) or a face (in 3D) will be
        considered neighbors. A connectivity of `label_img.ndim` means
        pixels sharing a corner will be considered neighbors.
    mode : string in {'thick', 'inner', 'outer', 'subpixel'}
        How to mark the boundaries:
        - thick: any pixel not completely surrounded by pixels of the
          same label (defined by `connectivity`) is marked as a boundary.
          This results in boundaries that are 2 pixels thick.
        - inner: outline the pixels *just inside* of objects, leaving
          background pixels untouched.
        - outer: outline pixels in the background around object
          boundaries. When two objects touch, their boundary is also
          marked.
        - subpixel: return a doubled image, with pixels *between* the
          original pixels marked as boundary where appropriate.
    Returns
    -------
    boundaries : array of bool, same shape as `label_img`
        A bool image where ``True`` represents a boundary pixel. For
        `mode` equal to 'subpixel', ``boundaries.shape[i]`` is equal
        to ``2 * label_img.shape[i] - 1`` for all ``i`` (a pixel is
        inserted in between all other pairs of pixels).

    """
    connectivity = 1
    if binary_img.shape[2] == 1:
        binary_img = binary_img[..., 0]
    result = np.zeros(binary_img.shape, bool)
    binary_img = np.array(binary_img, bool)

    trimed_idx = one_roi(binary_img, margin=2, return_index=True)
    binary_img = binary_img[trimed_idx]

    ndim = binary_img.ndim
    footprint = generate_binary_structure(ndim, connectivity)

    if mode == "inner":
        ero = binary_erosion(binary_img, footprint)
        boundaries = binary_img & (~ero)
    elif mode == "outer":
        dil = binary_dilation(binary_img, footprint)
        boundaries = (~binary_img) & dil
    elif mode == "thick":
        dil = binary_dilation(binary_img, footprint)
        ero = binary_erosion(binary_img, footprint)
        boundaries = dil ^ ero
    else:
        raise Exception(f"not supported mode {mode}")
    result[trimed_idx] = boundaries
    if result.ndim == 2:
        result = result.reshape(result.shape[0], result.shape[1], 1)
    return result
