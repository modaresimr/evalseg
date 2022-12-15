import numpy as np
from scipy.ndimage import distance_transform_edt

from ..common import Cache


# @Cache.memoize()
def expand_labels(label_image, spacing=None, mask_roi=None):
    """Expand labels in label image by ``distance`` pixels without overlapping.

    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).

    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    spacing: iterable of floats, optional
        Spacing between voxels in each spatial dimension. If None, then the spacing between pixels/voxels in each dimension is assumed 1.

    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged

    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.

    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.

    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.

    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`

    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559

    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])

    Labels will not overwrite each other:

    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])

    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.

    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """
    res = np.zeros(label_image.shape, np.uint8)
    mask_roi = np.s_[:] if mask_roi is None else mask_roi

    label_image = label_image[mask_roi]
    nearest_label_coords = distance_transform_edt(
        label_image == 0,
        return_distances=False,
        return_indices=True,
        sampling=spacing,
    )

    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    res[mask_roi] = nearest_labels
    return res


