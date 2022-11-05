

import numpy as np
import PIL
from PIL import Image


def concat(list_im, out=None, dir='h'):
    """
    concat multiple images 
    """

    imgs = [Image.open(i) for i in list_im]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    shape = sorted([(np.sum(i.size), i.size) for i in imgs])[-1][1]
    if dir == 'h':
        imgs_comb = np.hstack([i.resize(np.multiply(i.size, shape[1]/i.size[1]).astype(int)) for i in imgs])
    else:
        imgs_comb = np.vstack([i.resize(np.multiply(i.size, shape[0]/i.size[0]).astype(int)) for i in imgs])

    # save that beautiful picture
    imgs_comb = Image.fromarray(imgs_comb)
    if out:
        imgs_comb.save(out)
    return imgs_comb
