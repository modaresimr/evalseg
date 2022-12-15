
from skimage.morphology import remove_small_holes
from . import connected_components


def find_holes(image):
    total_area = image.sum()
    without_holes = remove_small_holes(image, total_area)
    holes = without_holes ^ image
    labels, gN = connected_components(holes, return_N=True)

    return labels
