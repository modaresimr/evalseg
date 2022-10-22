import numpy as np


def one_roi(img, margin=0, *, ignore=[], threshold=10, return_index=False):
    allzeros = np.where((img < threshold) & (img != 0))

    if type(margin) == int:
        margin = [margin for _ in range(img.ndim)]
    idx = ()
    for i in range(len(allzeros)):
        if i in ignore or len(allzeros[i]) == 0:
            idx += (np.s_[:],)
        else:
            idx += (
                np.s_[
                    max(0, allzeros[i].min() - margin[i]) : min(
                        img.shape[i], allzeros[i].max() + margin[i] + 1
                    )
                ],
            )
    if return_index:
        return idx
    return img[idx]
