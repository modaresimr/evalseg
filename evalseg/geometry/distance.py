import edt
import numpy as np

from .roi import one_roi


def distance(img, spacing=None, mode="in", mask_roi=None):
    """
    mode=in,out,both
    """
    spacing = spacing if not (spacing is None) else [1, 1, 1]
    trimed_idx = mask_roi if mask_roi is not None else np.s_[:, :, :]

    if img.shape[2] == 1 or img.ndim == 2:
        img = img.reshape(img.shape[0], img.shape[1])
        trimed_idx = trimed_idx[0], trimed_idx[1]
        spacing = spacing[0], spacing[1]

    orig_img = img

    # if mask_roi ==None and not (mask is None):
    #     trimed_idx = one_roi(mask, margin=4, return_index=True)
    #     mask_roi = trimed_idx

    # trimed_idx = one_roi(img, margin=(np.array(img.shape) * ignore_distance_rate + 4).astype(int),
    #                      return_index=True, mask_roi=mask_roi)

    modes = ["in", "out"] if mode == "both" else [mode]

    dst = np.zeros(orig_img.shape, np.float16)

    if "out" in modes:
        # print('shape', img.shape, dst.shape)
        imgo = img[trimed_idx]
        newdst = edt.edt(~imgo, anisotropy=spacing, black_border=False)
        dst[:] = newdst.max()
        dst[trimed_idx] = newdst
    if "in" in modes:
        new_trimed_idx = one_roi(orig_img, mask_roi=trimed_idx, margin=4, return_index=True)
        imgi = img[new_trimed_idx]
        dst[new_trimed_idx] += edt.edt(imgi, anisotropy=spacing, black_border=True) * (-1 if "out" in modes else 1)

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
