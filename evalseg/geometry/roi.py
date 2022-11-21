import numpy as np


def calc_safe_roi(shape, data_roi, margin=0, roi_rate=5):
    idx = ()
    for i in range(len(shape)):

        s = data_roi[i].start or 0
        e = data_roi[i].stop or shape[i]
        new_margin = margin+(e-s)*roi_rate
        idx += (np.s_[max(0, s - new_margin):min(shape[i], e + new_margin)],)
    return idx


def one_roi(img, margin=0, *, ignore=[], return_index=False, mask_roi=None, fill_value=None):
    s = [0 for i in range(img.ndim)]
    if mask_roi is not None:
        img = img[mask_roi]
        s = [m.start for m in mask_roi]
    fill_value = fill_value if fill_value is not None else False if img.dtype == bool else 0

    # if threshold:
    #     allzeros = np.where((img < threshold) & (img != fill_value))
    # else:
    allzeros = np.where(img != fill_value)

    if type(margin) == int:
        margin = [margin for _ in range(img.ndim)]
    idx = ()
    for i in range(len(allzeros)):
        if i in ignore:
            idx += (np.s_[:],)
        elif len(allzeros[i]) == 0:
            idx += (np.s_[0:0],)
        else:
            idx += (np.s_[max(0, allzeros[i].min() - margin[i])+s[i]:
                          min(img.shape[i], allzeros[i].max() + margin[i] + 1)+s[i]],)
    if return_index:
        return idx
    return img[idx]
