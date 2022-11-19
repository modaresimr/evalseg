import numpy as np


def one_roi(img, margin=0, *, ignore=[], threshold=None, return_index=False, mask_roi=None):
    s = [0 for i in range(img.ndim)]
    if mask_roi is not None:
        img = img[mask_roi]
        s = [m.start for m in mask_roi]

    if threshold:
        allzeros = np.where((img < threshold) & (img != 0))
    else:
        allzeros = np.where(img != 0)

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
