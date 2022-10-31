import cv2
import numpy as np

epsilon = 0.0001


def clahe(img):
    img = img.reshape(img.shape[0], img.shape[1], -1).copy()
    imax = img.max()
    imin = img.min()
    normalimg = (img - imin) / (imax - imin + epsilon)

    clahe = cv2.createCLAHE(clipLimit=5.0)
    for i in range(img.shape[2]):
        img[:, :, i] = clahe.apply(np.uint8(normalimg[:, :, i] * 255))

    return img / 255 * (imax - imin) + imin
