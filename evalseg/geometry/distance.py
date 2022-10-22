import edt
import numpy as np

from .roi import one_roi


def distance(img, spacing=None, mode="in", mask=None, ignore_distance_rate=1):
    """
    mode=in,out,both
    """
    spacing = spacing if not (spacing is None) else [1, 1, 1]
    if img.shape[2] == 1:
        img = img.reshape(img.shape[0], img.shape[1])
        if mask is not None:
            mask = mask.reshape(mask.shape[0], mask.shape[1])
        spacing = spacing[0], spacing[1]

    orig_img = img

    if not (mask is None):
        trimed_idx = one_roi(mask, margin=4, return_index=True)
    else:
        trimed_idx = one_roi(
            img,
            margin=(np.array(img.shape) * ignore_distance_rate + 4).astype(
                int
            ),
            return_index=True,
        )

        # trimed_idx = np.s_[:, :, :]

    img = img[trimed_idx]
    modes = ["in", "out"] if mode == "both" else [mode]

    dst = np.zeros(orig_img.shape, np.float16)
    if "out" in modes:
        dst[trimed_idx] = edt.edt(~img, anisotropy=spacing, black_border=False)
    if "in" in modes:
        trimed_idx = one_roi(img, margin=4, return_index=True)
        img = img[trimed_idx]
        dst[trimed_idx] += edt.edt(
            img, anisotropy=spacing, black_border=True
        ) * (-1 if "out" in modes else 1)

    # if mode == 'both':
    #     dst[trimed_idx] = edt.edt(~img, anisotropy=spacing, black_border=False)

    #     trimed_idx = one_roi(img, margin=4, return_index=True)
    #     img = img[trimed_idx]
    #     dst[trimed_idx] -= edt.edt(img, anisotropy=spacing, black_border=True)
    # elif mode == 'in':
    #     trimed_idx = one_roi(img, margin=4, return_index=True)
    #     img = img[trimed_idx]
    #     dst[trimed_idx] = edt.edt(img, anisotropy=spacing, black_border=True)
    # else:
    #     dst[trimed_idx] = edt.edt(~img, anisotropy=spacing, black_border=False)

    if img.ndim == 2:
        dst = dst.reshape(dst.shape[0], dst.shape[1], -1)
    return dst
