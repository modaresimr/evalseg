import numpy as np
import scipy
import skimage

from ..common import Cache
from .roi import one_roi


# @Cache.memoize()
def skeletonize(img, spacing=None, surface=False, do_smoothing=True):

    if spacing is None:
        spacing = [1, 1, 1]
    orig_img = img
    trimed_idx = one_roi(img, margin=2, return_index=True)
    # trimed_idx = np.s_[:, :, :]
    img = img[trimed_idx]

    skel = np.zeros(orig_img.shape)
    spacing = np.array(spacing)
    spacing = spacing / spacing.min()

    img2s = (
        skimage.transform.rescale(
            img, spacing, preserve_range=True, mode="edge"
        )
        > 0
    )

    if do_smoothing:
        img2s = scipy.ndimage.median_filter(img2s, 5)

    if surface:
        skel2 = skimage.morphology.medial_surface(img2s) > 0
    else:
        skel2 = skimage.morphology.skeletonize_3d(img2s) > 0
    skel[trimed_idx] = (
        skimage.transform.resize(
            skel2 * 1, img.shape, preserve_range=True, mode="edge"
        )
        > 0
    )
    if skel.sum() == 0:
        cx, cy, cz = scipy.ndimage.center_of_mass(orig_img)
        skel[int(cx), int(cy), int(cz)] = 1
    return skel
