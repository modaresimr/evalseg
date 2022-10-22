import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .. import ct, geometry

epsilon = 0.0001


def multi_plot_2d(img, pred, dst=None, spacing=None, args={}):
    spacing = np.array([1, 1, 1] if spacing is None else spacing)
    f = {}
    imglbl = args.get("imglabel", "img")
    origsize_lbl = "orig_size"
    if args.get("add_notzoom_img", 0):
        origsize_lbl = imglbl
        imglbl = "Zoom to ROI"
    gtlbl = "GroundTruth"

    items = {imglbl: img, **pred}

    if args.get("clahe", 0):
        items[imglbl] = ct.clahe(items[imglbl])

    if args.get("crop2roi", 0):
        roi = ct.ct_roi(items[imglbl], True)
        items = {p: items[p][roi] for p in items}

    if args.get("zoom2segments", 0):
        notzoom_img = items[imglbl]

        orig_ratio = notzoom_img.shape[1] / notzoom_img.shape[0]
        zoom_roi = ct.segment_roi(
            [items[p] for p in items if p != imglbl],
            wh_ratio=orig_ratio,
            mindim=[20 / spacing[0], 20 / spacing[1], -1],
        )
        items = {p: items[p][zoom_roi] for p in items}
        if args.get("add_notzoom_img", 0):
            # items={p:CTHelper.upscale_ct(items[p],notzoom_img.shape) for p in items}
            items = {origsize_lbl: notzoom_img, **items}

    #         if origsize_lbl in items:
    #             items[origsize_lbl]=CTHelper.claheCT(items[origsize_lbl])

    img = items[imglbl]

    gt = items[gtlbl]
    normalimg = (img - img.min()) / (img.max() - img.min() + epsilon)

    data = {}
    for p in tqdm(items, leave=False):
        x = items[p]
        clipmin = x.min()
        clipmax = x.max()

        data[p] = {
            "pred": (np.clip(x, clipmin, clipmax) - clipmin)
            / (clipmax - clipmin + 0.0000000001)
        }  # min(clipmax,items[p].max())}

        if p == origsize_lbl:
            pass
        elif p != imglbl:
            fp = (gt != x) & (x > 0)
            fn = (gt != x) & (gt > 0)
            tp = (gt == x) & (gt > 0)
            data[p]["fp"] = fp
            data[p]["fn"] = fn
            data[p]["tp"] = tp

    mri_cmap = "bone"  # plotui.customMRIColorMapForMPL_TPFPFN()

    col = min(len(items), 5)
    row = (len(items) - 1) // col + 1
    zs = items[imglbl].shape[2]

    z_titles = args.get("z_titles", [i for i in range(zs)])
    for anim in range(zs):

        # fig, ax1 = plt.subplots(1, 1, figsize=(row, col),dpi=100)
        # ,gridspec_kw={'left':0, 'right':0, 'top':0, 'bottom':0}
        fig, axes = plt.subplots(row, col, figsize=(col * 2, row * 2), dpi=100)
        aspect = spacing[0] / spacing[1]
        fig.suptitle(f"frame: {z_titles[anim]}")
        axes = axes.reshape(-1)
        for i, p in enumerate(data):
            current = {d: data[p][d][:, :, anim] for d in data[p]}

            imgsize = 20
            if p in [imglbl, origsize_lbl]:
                axes[i].imshow(
                    current["pred"],
                    cmap=mri_cmap,
                    vmin=0,
                    vmax=1,
                    alpha=1,
                    interpolation="nearest",
                    aspect=aspect,
                )
                if p == origsize_lbl:
                    from matplotlib.patches import Rectangle

                    y, x = zoom_roi[0].start, zoom_roi[1].start
                    h, w = zoom_roi[0].stop - y, zoom_roi[1].stop - x

                    axes[i].add_patch(
                        Rectangle(
                            (x, y),
                            w,
                            h,
                            facecolor="none",
                            edgecolor="blue",
                            lw=2,
                        )
                    )
            else:
                from matplotlib.colors import (
                    LinearSegmentedColormap,
                    ListedColormap,
                )

                if args.get("add_backimg", 0) and not ():
                    axes[i].imshow(
                        normalimg[:, :, anim] / 2,
                        cmap=mri_cmap,
                        vmin=0,
                        vmax=1,
                        alpha=1,
                        interpolation="nearest",
                        aspect=aspect,
                    )
                if p != gtlbl:
                    if args.get("show_tp_fp_fn", 1):
                        if "tp" in current and current["tp"].sum() > 0:
                            # axes[i].contour(current['tp'],corner_mask=False,cmap=ListedColormap([(0,0,0,0),'lime']),vmin=0, vmax=1, alpha=1 )
                            cmap = ListedColormap([(0, 0, 0, 0), "lime"])
                            axes[i].imshow(
                                current["tp"],
                                cmap=cmap,
                                vmin=0,
                                vmax=1,
                                alpha=1,
                                aspect=aspect,
                            )
                        if "fp" in current and current["fp"].sum() > 0:
                            cmap = ListedColormap([(0, 0, 0, 0), "yellow"])
                            axes[i].imshow(
                                current["fp"],
                                cmap=cmap,
                                vmin=0,
                                vmax=1,
                                alpha=1,
                                aspect=aspect,
                            )
                            axes[i].contour(
                                current["fp"],
                                corner_mask=False,
                                cmap=cmap,
                                vmin=0,
                                vmax=1,
                                alpha=1,
                            )
                        if "fn" in current and current["fn"].sum() > 0:
                            cmap = ListedColormap([(0, 0, 0, 0), "red"])
                            axes[i].imshow(
                                current["fn"],
                                cmap=cmap,
                                vmin=0,
                                vmax=1,
                                alpha=1,
                                aspect=aspect,
                            )
                            axes[i].contour(
                                current["fn"],
                                corner_mask=False,
                                cmap=cmap,
                                vmin=0,
                                vmax=1,
                            )
                    else:
                        cmap = ListedColormap([(0, 0, 0, 0), "lime"])
                        axes[i].imshow(
                            data[gtlbl]["pred"],
                            cmap=cmap,
                            vmin=0,
                            vmax=1,
                            alpha=0.5,
                            aspect=aspect,
                        )

                if current["pred"].sum() > 0:
                    color = "lime" if p == gtlbl else "yellow"
                    axes[i].contour(current["pred"], colors=color, alpha=1)

            axes[i].set_axis_off()
            axes[i].set_ylim(0, current["pred"].shape[0])
            axes[i].set_xlim(0, current["pred"].shape[1])
            axes[i].set_title(p)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        if dst:
            fig.savefig(f"{dst}{anim}.png")
        if args.get("show", 1):
            fig.show()
        else:
            plt.close()
