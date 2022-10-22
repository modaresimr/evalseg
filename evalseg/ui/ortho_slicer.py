import k3d
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .. import ct, geometry


def ortho_slicer(img, pred, cut, spacing=None, args={}):
    spacing = np.array([1, 1, 1] if spacing is None else spacing)
    row = len(pred)
    col = 3
    fig, axes = plt.subplots(row, col, figsize=(col * 2, row * 2), dpi=100)
    if row == 1:
        axes = [axes]

    if len(pred) == 0:
        pred["data"] = np.zeros(img.shape)

    for pi, p in enumerate(pred):
        for i in range(3):
            cut_img, cur_spa = geometry.slice(img, spacing, i, cut[i])

            axes[pi][i].imshow(
                ct.clahe(cut_img), cmap="bone", aspect=cur_spa[0] / cur_spa[1]
            )
            predcut, dim_new = geometry.slice(
                pred[p], np.array([0, 1, 2]), i, cut[i]
            )

            # axes[pi][i].imshow(predcut, cmap='bone')

            if predcut.sum() > 0:
                axes[pi][i].contour(predcut, cmap="bone")

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
    if args.get("dst", ""):
        fig.savefig(args["dst"] + ".png")
    if args.get("show", 1):
        fig.show()
    else:
        plt.close()
