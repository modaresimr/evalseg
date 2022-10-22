from collections import defaultdict

import k3d
import numpy as np
from k3d.colormaps import basic_color_maps, matplotlib_color_maps
from tqdm.auto import tqdm

from .. import common, geometry


def colormap_convert(arr):
    arr = np.multiply(arr, 255).astype(int)
    tall = [
        (arr[x] << 24) + (arr[x + 1] << 16) + (arr[x + 2] << 8) + (arr[x + 3])
        for x in range(0, len(arr), 4)
    ]
    # if indx==None:
    return common.CircleList(tall)
    # return [tall[indx%len(tall)]])


# colormap=cm(basic_color_maps.RainbowDesaturated)
cls_colormap = colormap_convert(matplotlib_color_maps.tab10)[::28]
tp_colormap = colormap_convert(matplotlib_color_maps.Greens)[28 * 3 :: 28]
fn_colormap = colormap_convert(matplotlib_color_maps.Blues)[28 * 3 :: 28]
fp_colormap = colormap_convert(matplotlib_color_maps.Reds)[28 * 3 :: 28]

bone_colormap = colormap_convert(matplotlib_color_maps.bone)


def multi_plot(gt, preds, spacing=None, args={}):
    spacing = np.array([1, 1, 1] if spacing is None else spacing)
    gt = gt.reshape(gt.shape[0], gt.shape[1], -1)
    args = defaultdict(str, args)
    preds = {
        pr: preds[pr].reshape(preds[pr].shape[0], preds[pr].shape[1], -1)
        for pr in preds
    }

    hasz = gt.shape[2] > 2

    plot = k3d.plot(
        grid=[0, gt.shape[0], 0, gt.shape[1], 0, gt.shape[2]],
        name=args["dst"],
        grid_auto_fit=False,
        camera_auto_fit=hasz,
    )
    if args["show"]:
        plot.display()
    if not hasz:
        plot.camera = [512, 200, 200, 0, 200, 200, 0, 0, 1]

    v = k3d.voxels(
        gt.astype(np.uint8),
        opacity=0.3,
        compression_level=9,
        name="gt",
        group="gt",
        scaling=spacing,
    )
    v.visible = args.get("show_all_gt", 0)
    v.outlines = False
    plot += v
    # for c in range(1, pred.max() + 1):
    #     plot += k3d.voxels((gt == c).astype(np.uint8), opacity=0.0, color=color_map[c], compression_level=9, name=f'{c}', group='gt')
    for c in tqdm(range(1, int(gt.max() + 1)), leave=False):
        v = k3d.voxels(
            (gt == c).astype(np.uint8),
            opacity=0.2,
            color_map=[cls_colormap[c]],
            compression_level=9,
            name=f"{c}",
            group="gt",
        )
        v.visible = args.get("show_each_gt", {c: 1}).get(c, 0)
        v.outlines = False
        plot += v
        if args.get("calc_skeleton_gt", {}).get(c, 0):
            shape = (gt == c).astype(np.uint8)
            skeleton = geometry.skeletonize(shape, spacing=spacing)
            #         skeleton = skeletonize(shape)

            v = k3d.voxels(
                skeleton.astype(np.uint8),
                opacity=1,
                color_map=[cls_colormap[c]],
                compression_level=1,
                name=f"sk-{c}",
                group="gt",
            )

            v.visible = args.get("show_skeleton_gt", {}).get(c, 0)
            v.outlines = True
            plot += v

            v2 = k3d.voxels(
                skeleton.astype(np.uint8),
                opacity=0.2,
                color_map=[cls_colormap[c]],
                compression_level=1,
                name=f"smooth-{c}",
                group="gt",
                scaling=spacing,
            )
            v2.visible = args.get("show_smooth_gt", {}).get(c, 0)
            v2.outlines = False
            plot += v2

    for p in tqdm(preds, leave=False):
        pred = preds[p].astype(np.uint8)
        fp = (gt != pred) & (pred > 0)
        fn = (gt != pred) & (gt > 0)
        tp = (gt == pred) & (gt > 0)

        v = k3d.voxels(
            fp.astype(np.uint8),
            opacity=0.6,
            color_map=fp_colormap,
            compression_level=9,
            name="fp",
            group=p,
            scaling=spacing,
        )
        v.visible = args.get("show_all_fp", 0)
        v.outlines = False
        plot += v

        v = k3d.voxels(
            fn.astype(np.uint8),
            opacity=0.6,
            color_map=fn_colormap,
            compression_level=9,
            name="fn",
            group=p,
            scaling=spacing,
        )  #
        v.visible = args.get("show_all_fn", 0)
        v.outlines = False
        plot += v

        v = k3d.voxels(
            tp.astype(np.uint8),
            opacity=0.2,
            color_map=tp_colormap,
            compression_level=9,
            name="tp",
            group=p,
            scaling=spacing,
        )
        v.visible = args.get("show_all_tp", 0)
        v.outlines = False
        plot += v

        v = k3d.voxels(
            pred.astype(np.uint8),
            opacity=0.2,
            compression_level=9,
            color_map=cls_colormap,
            name="pred",
            group=p,
            scaling=spacing,
        )
        v.visible = args.get("show_all_pred", 0)
        v.outlines = False
        plot += v

        for c in tqdm(range(1, int(pred.max() + 1)), leave=False):
            v = k3d.voxels(
                (pred == c).astype(np.uint8),
                opacity=0.2,
                color_map=[cls_colormap[c]],
                compression_level=9,
                name=f"{c}",
                group=p,
                scaling=spacing,
            )
            v.visible = args.get("show_each_pred", {c: 1}).get(c, 0)
            v.outlines = False
            plot += v

        # plot += k3d.voxels(pred * 3, opacity=0.3, compression_level=5, name='pred', group=p)
        # plot += k3d.voxels(pred * 3, opacity=0.3, compression_level=5, name=p)
        # for c in range(1, pred.max() + 1):
        #     plot += k3d.voxels((pred == c).astype(np.uint8), opacity=0.0, color=color_map[c + 5], compression_level=9, name=f'{c}', group=p)

    if args["dst"]:
        with open(args["dst"], "w") as fp:
            fp.write(plot.get_snapshot())


def plot_voxels(gt, dst=None, show=False):

    plot = k3d.plot(
        grid=[0, 512, 0, 512, 0, 512], name=dst, grid_auto_fit=False
    )

    plot += k3d.voxels(gt.astype(np.uint8), opacity=0.3, compression_level=9)
    # for c in range(1, pred.max() + 1):
    #     plot += k3d.voxels((gt == c).astype(np.uint8), opacity=0.0, color=color_map[c], compression_level=9, name=f'{c}', group='gt')

    if show:
        plot.display()
    if dst != None:
        with open(dst, "w") as fp:
            fp.write(plot.get_snapshot())
