import k3d
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .. import ct_helper, geometry, io


def ortho_slicer(img, pred, cut, spacing=None, show=True, dst=None):
    spacing = np.array([1, 1, 1] if spacing is None else spacing)
    row = len(pred)
    col = 3
    fig, axes = plt.subplots(row, col, figsize=(col * 2, row * 2), dpi=100)
    if row == 1:
        axes = [axes]

    if len(pred) == 0:
        pred["data"] = np.zeros_like(img)

    for pi, p in enumerate(pred):
        for i in range(3):
            cut_img, cur_spa = geometry.slice(img, spacing, i, cut[i])

            axes[pi][i].imshow(ct_helper.clahe(cut_img), cmap="bone", aspect=cur_spa[0] / cur_spa[1])
            predcut, dim_new = geometry.slice(pred[p], np.array([0, 1, 2]), i, cut[i])

            # axes[pi][i].imshow(predcut, cmap='bone')

            if predcut.sum() > 0:
                axes[pi][i].contour(predcut)

            axes[pi][i].axhline(cut[dim_new[0]])
            axes[pi][i].axvline(cut[dim_new[1]])
            # if i == 0:
            #     axes[pi][i].axhline(cut[2])
            #     axes[pi][i].axvline(cut[1])
            # if i == 1:
            #     axes[pi][i].axhline(cut[2])
            #     axes[pi][i].axvline(cut[0])
            # if i == 2:
            #     axes[pi][i].axhline(cut[1])
            #     axes[pi][i].axvline(cut[0])
            # axes[i].invert_xaxis()
            axes[pi][i].set_ylim(0, predcut.shape[0])
            axes[pi][i].set_xlim(0, predcut.shape[1])
            # axes[i].invert_yaxis()
        axes[pi][1].set_title(f"{p} {cut}")
    if dst:
        fig.savefig(dst + ".png")
    if show:
        fig.show()
    else:
        plt.close()


def ortho_slicer_segment(img, preds, cut, show=True, dst=None):
    row = len(preds)
    col = 3
    fig, axes = plt.subplots(row, col, figsize=(col * 2, row * 2), dpi=100)
    if row == 1:
        axes = [axes]

    if len(preds) == 0:
        preds["data"] = io.SegmentArray(np.zeros_like(img))

    for pi, p in enumerate(preds):
        for i in range(3):
            cut_img = geometry.slice_segment(img, i, cut[i])

            axes[pi][i].imshow(ct_helper.clahe(cut_img.todense()), cmap="bone", aspect=cut_img.voxelsize[0] / cut_img.voxelsize[1])
            predcut = geometry.slice_segment(preds[p], i, cut[i],spacing=np.array([0, 1, 2]))


            # axes[pi][i].imshow(predcut, cmap='bone')

            if predcut.sum() > 0:
                axes[pi][i].contour(predcut.todense())

            axes[pi][i].axhline(cut[predcut.voxelsize[0]])
            axes[pi][i].axvline(cut[predcut.voxelsize[1]])

            axes[pi][i].set_ylim(0, predcut.shape[0])
            axes[pi][i].set_xlim(0, predcut.shape[1])

        axes[pi][1].set_title(f"{p} {cut}")
    if dst:
        fig.savefig(dst + ".png")
    if show:
        fig.show()
    else:
        plt.close()
