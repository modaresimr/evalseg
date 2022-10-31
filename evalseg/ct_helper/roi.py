import cv2
import numpy as np

epsilon = 0.0001


def ct_roi(img, keepRatio=True):

    contours = []
    for i in range(img.shape[2]):
        contours = [*contours, *_get_contours(img[:, :, i])]
    bounds = _find_boundaries(img, contours)
    roi = np.array([[bounds[1], bounds[3]], [bounds[0], bounds[2]]])
    if img.sum() < 100:
        roi = np.array([[0, img.shape[1]], [0, img.shape[0]]])
    if keepRatio:
        _extend_roi_to_ratio(img.shape, roi, img.shape[1] / img.shape[0])
    idx = ()
    for i in range(len(roi)):
        idx += (np.s_[roi[i][0] : roi[i][1] + 1],)
    # print(idx)
    return idx


def segment_roi(segments, margin=10, wh_ratio=1, mindim=[50, 50, -1]):

    imgshape = segments[0].shape
    roi = np.zeros((len(imgshape), 2), int)
    roi[:, 0] = 100000

    for seg in segments:
        nonzero = np.where(seg != 0)

        for i in range(len(nonzero)):
            if len(nonzero[i]) > 0:
                roi[i][0] = max(0, min(nonzero[i].min() - margin, roi[i][0]))
                roi[i][1] = min(
                    imgshape[i], max(nonzero[i].max() + margin + 1, roi[i][1])
                )

    for i in range(len(roi)):
        if roi[i][1] < roi[i][0]:
            roi[i] = [0, imgshape[i]]

    _extend_roi_shape(
        imgshape,
        roi,
        [mindim[i] - (roi[i][1] - roi[i][0]) for i in range(len(roi))],
    )
    _extend_roi_to_ratio(imgshape, roi, wh_ratio)

    idx = ()
    for i in range(len(roi)):
        idx += (np.s_[roi[i][0] : roi[i][1] + 1],)
    # print(idx)
    return idx


def _extend_roi_shape(imgshape, roi, shape):
    for dim in range(len(shape)):
        if shape[dim] <= 0:
            continue
        roi[dim][0] -= shape[dim] / 2
        roi[dim][1] += shape[dim] / 2
        if roi[dim][0] < 0:
            roi[dim][:] -= roi[dim][0]
        elif roi[dim][1] > imgshape[dim]:
            roi[dim][:] -= roi[dim][1] - imgshape[dim]
        roi[dim][:] = roi[dim][:].clip(0, imgshape[dim])
    return roi


def _extend_roi_to_ratio(imgshape, roi, wh_ratio):
    w = roi[0][1] - roi[0][0]
    h = roi[1][1] - roi[1][0]

    nh = w * wh_ratio
    nw = h / wh_ratio
    _extend_roi_shape(imgshape, roi, [nw - w, nh - h])


def _normalizeImage(img):
    return (img - img.min()) / (img.max() - img.min() + epsilon)


def _get_contours(img):
    img = np.uint8(_normalizeImage(img) * 255)
    kernel = np.ones((3, 3), np.float32) / 9
    img = cv2.filter2D(img, -1, kernel)

    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, 2, 1)

    # filter contours that are too large or small
    ih, iw = img.shape
    totalArea = ih * iw
    contours2 = contours
    for mx in np.arange(0.5, 1, 0.1):
        tmp = [cc for cc in contours if _contourOK(cc, totalArea, mx)]
        if len(tmp) != 0:
            contours2 = tmp
            break
    return contours2


def _contourOK(cc, totalArea, ignore_max=0.9):
    x, y, w, h = cv2.boundingRect(cc)
    if (w < 50 and h > 150) or (w > 150 and h < 50):
        return False  # too narrow or wide is bad
    area = cv2.contourArea(cc)
    # print(f'area={area}')
    return (area < totalArea * ignore_max) & (area > totalArea * 0.2)


def _find_boundaries(img, contours):
    # margin is the minimum distance from the edges of the image, as a fraction
    ih, iw = img.shape[0], img.shape[1]
    minx = iw
    miny = ih
    maxx = 0
    maxy = 0

    for cc in contours:
        x, y, w, h = cv2.boundingRect(cc)
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x + w > maxx:
            maxx = x + w
        if y + h > maxy:
            maxy = y + h

    return (minx, miny, maxx, maxy)
