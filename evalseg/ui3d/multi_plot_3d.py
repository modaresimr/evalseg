import k3d
import numpy as np
from . import k3d_subplots, k3d_tools
from .. import ct_helper
from ipywidgets import HBox
from IPython.display import display
import skimage


def multi_plot_3d(ct, gt, preds, *, dst=None, spacing=None, ctlbl='CT', show_orig_size_ct=True,
                  show_tp_fp_fn=True, zoom2segments=True,
                  col=5, show=True, z_titles=[], gtlbl="GroundTruth"):
    hasz = 0 and gt.ndim == 3 and gt.shape[2] > 1
    subplots = k3d_subplots()
    scaledown = (5, 5, 5)
    ct = skimage.transform.downscale_local_mean(ct, scaledown)
    gt = skimage.transform.downscale_local_mean(gt, scaledown)
    preds = {p: skimage.transform.downscale_local_mean(preds[p], scaledown) for p in preds}

    gtmax = (gt.sum(axis=2).sum(axis=1).argmax(), gt.sum(axis=2).sum(axis=0).argmax(), gt.sum(axis=1).sum(axis=0).argmax())
    spacing = [spacing[0], spacing[1], spacing[2]]
    gto = gt
    gt = gt.copy()

    zoom_roi = np.s_[0:gt.shape[0], 0:gt.shape[1], 0:gt.shape[2]]
    preds = {p: preds[p].copy() for p in preds}
    if zoom2segments:
        zoom_roi = ct_helper.segment_roi(
            [gt, *list(preds.values())],
            mindim=[20 / spacing[0], 20 / spacing[1], -1],
        )
        rate = max([(zoom_roi[i].stop-zoom_roi[i].start)/gt.shape[i] for i in range(3)])
        zoom_roi = ct_helper.segment_roi(
            [gt, *list(preds.values())],
            mindim=[gt.shape[i]*rate for i in range(3)],
        )

        gt = gt[zoom_roi]
        preds = {p: preds[p][zoom_roi] for p in preds}

    if ct is not None:
        ct = ct.copy()

        plot_ct = k3d.plot(grid=get_bound(ct), name=ctlbl, grid_auto_fit=False, camera_auto_fit=hasz)
        subplots.add_plot(plot_ct, ctlbl)
        # .display()
        # ct[gtmax[0]:, gtmax[1]:, gtmax[2]:] = -1000
        # ct[gto > 0] = 1000.

        # plot_ct += k3d.volume(ct.astype(np.float32),
        #                       alpha_coef=100,
        #                       color_map=k3d.colormaps.matplotlib_color_maps.Turbo,
        #                       #   color_map=k3d.colormaps.paraview_color_maps.Nic_Edge,
        #                       #   color_map=k3d.colormaps.matplotlib_color_maps.gist_rainbow,
        #                       scaling=spacing,
        #                       color_range=[100, 1100],
        #                       bounds=ct_bounds,
        #                       #   gradient_step=1,
        #                       name='ct')
        origin = np.array([zoom_roi[i].start for i in range(3)])
        size = np.array([zoom_roi[i].stop-zoom_roi[i].start for i in range(3)])
        cube = k3d_tools.Cube(origin, size)
        # print('origin', [zoom_roi[i].start for i in range(3)])
        # print('size', [zoom_roi[i].stop-zoom_roi[i].start for i in range(3)])
        # print(cube.vertices)

        # plot_ct += k3d.mesh(cube.vertices, cube.indices, wireframe=True, scaling=spacing,)
        # plot_ct += k3d.points(cube.vertices, point_size=0.2, shader='3d', scaling=spacing, color=0x3f6bc5)
        plot_ct += k3d.mip(ct.astype(np.float32),

                           color_map=k3d.colormaps.matplotlib_color_maps.Turbo,
                           #   color_map=k3d.colormaps.paraview_color_maps.Nic_Edge,
                           #   color_map=k3d.colormaps.matplotlib_color_maps.gist_rainbow,
                           scaling=spacing,
                           color_range=[-200, 900],
                           bounds=ct_bounds,
                           samples=100,
                           #   gradient_step=1,
                           name='ct')
        plot_ct += k3d.mip(gto.copy().astype(np.float32),
                           #   alpha_coef=100,
                           color_map=k3d.colormaps.matplotlib_color_maps.Greys,
                           scaling=spacing,
                           color_range=[0, 1],
                           bounds=ct_bounds,
                           samples=100,
                           #   gradient_step=1,
                           name='ct-gt')

        plot_ct.camera = [ct.shape[0]/2, ct.shape[1], -ct.shape[2], ct.shape[0]/3, ct.shape[1]/3, ct.shape[2]/3, 1, 0, 0]

    bounds = [0, gt.shape[0], 0, gt.shape[1], 0, gt.shape[2]]
    plot_gt = k3d.plot(grid=bounds, name=ctlbl, grid_auto_fit=False, camera_auto_fit=hasz)
    # plot_gt.display()
    # plot_gt.camera = camera
    subplots.add_plot(plot_gt, gtlbl)
    plot_gt += k3d.volume(gt.astype(np.float32),
                          alpha_coef=100,
                          color_map=k3d.colormaps.matplotlib_color_maps.Greys,
                          scaling=spacing,
                          color_range=[0, 1],
                          bounds=bounds,
                          show_legend=False,
                          gradient_step=1,
                          name='gt')
    plot_gt.camera = [gt.shape[0]/2, gt.shape[1], -gt.shape[2], gt.shape[0]/3, gt.shape[1]/3, gt.shape[2]/3, 1, 0, 0]
    plot_gt.colorbar_object_id = -1
    plot_pr = {}
    for p in preds:

        plot_pred = k3d.plot(grid=bounds, name=ctlbl, grid_auto_fit=False, camera_auto_fit=hasz)
        subplots.add_plot(plot_pred, p)
        # plot_pr[p].display()
        # hbox.childern += (plot_pr[p],)
        # plot_pred += k3d.volume(preds[p].astype(np.float32),
        #                         alpha_coef=100,
        #                         color_map=k3d.colormaps.matplotlib_color_maps.Blues,
        #                         scaling=spacing,
        #                         color_range=[0, 1],
        #                         bounds=bounds,
        #                         show_legend=False,
        #                         gradient_step=1,
        #                         # translation=[0, 0, -gt.shape[0]],
        #                         name='pred')
        plot_pred.colorbar_object_id = -1
        tp = (gt > 0) & (preds[p] > 0)
        fp = (~(gt > 0)) & (preds[p] > 0)
        fn = ((gt > 0)) & ~(preds[p] > 0)
        plot_pred += k3d.volume(tp.astype(np.float32),
                                alpha_coef=100,
                                color_map=k3d.colormaps.matplotlib_color_maps.Greens,
                                scaling=spacing,
                                color_range=[0, 1],
                                bounds=bounds,
                                show_legend=False,
                                gradient_step=1,
                                # translation=[0, 0, -gt.shape[0]],
                                name='pred-tp')
        plot_pred += k3d.volume(fn.astype(np.float32),
                                alpha_coef=100,
                                color_map=k3d.colormaps.basic_color_maps.Gold,
                                scaling=spacing,
                                color_range=[0, 1],
                                bounds=bounds,
                                show_legend=False,
                                gradient_step=1,
                                # translation=[0, 0, -gt.shape[0]],
                                name='pred-fn')
        plot_pred += k3d.volume(fp.astype(np.float32),
                                alpha_coef=100,
                                color_map=k3d.colormaps.matplotlib_color_maps.Reds,
                                scaling=spacing,
                                color_range=[0, 1],
                                bounds=bounds,
                                show_legend=False,
                                gradient_step=1,
                                # translation=[0, 0, -gt.shape[0]],
                                name='pred-fp')

        plot_tp = k3d.plot(grid=bounds, name=f'{p}-tp', grid_auto_fit=False, camera_auto_fit=hasz)
        subplots.add_plot(plot_tp, f'{p} : True Positive')
        plot_tp += k3d.volume(tp.astype(np.float32),
                              alpha_coef=100,
                              color_map=k3d.colormaps.matplotlib_color_maps.Greens,
                              scaling=spacing,
                              color_range=[0, 1],
                              bounds=bounds,
                              show_legend=False,
                              gradient_step=1,
                              name=f'{p}-tp')
        plot_tp.colorbar_object_id = -1
        plot_fn = k3d.plot(grid=bounds, name=f'{p}-fn', grid_auto_fit=False, camera_auto_fit=hasz)
        subplots.add_plot(plot_fn, f'{p} : False Negative')
        plot_fn += k3d.volume(fn.astype(np.float32),
                              alpha_coef=100,
                              color_map=k3d.colormaps.basic_color_maps.Gold,
                              scaling=spacing,
                              color_range=[0, 1],
                              bounds=bounds,
                              show_legend=False,
                              gradient_step=1,
                              #   translation=[0, 0, gt.shape[0]],
                              name=f'{p}-fn')
        plot_fn.colorbar_object_id = -1
        plot_fp = k3d.plot(grid=bounds, name=f'{p}-fp', grid_auto_fit=False, camera_auto_fit=hasz)
        subplots.add_plot(plot_fp, f'{p} : False Positive')
        plot_fp += k3d.volume(fp.astype(np.float32),
                              alpha_coef=100,
                              color_map=k3d.colormaps.matplotlib_color_maps.Reds,
                              scaling=spacing,
                              color_range=[0, 1],
                              bounds=bounds,
                              gradient_step=1,
                              show_legend=False,
                              #   translation=[0, 0, gt.shape[0]],
                              name=f'{p}-fp')
        plot_fp.colorbar_object_id = -1
    subplots.sync_camera_view()
    return subplots


def plot_3d_old(ct, gt, pred, *, dst=None, spacing=None, ctlbl='CT', show_orig_size_ct=True,
                show_tp_fp_fn=True, zoom2segments=True,
                col=5, show=True, gtlbl="GroundTruth"):
    subplots = k3d_subplots()
    scaledown = (5, 5, 5)
    ct = skimage.transform.downscale_local_mean(ct, scaledown)
    gt = skimage.transform.downscale_local_mean(gt, scaledown)
    pred = skimage.transform.downscale_local_mean(pred, scaledown)

    spacing = np.array([spacing[0], spacing[1], spacing[2]])
    gto = gt
    gt = gt.copy()

    zoom_roi = np.s_[0:gt.shape[0], 0:gt.shape[1], 0:gt.shape[2]]
    pred = pred.copy()
    if zoom2segments and 0:
        zoom_roi = ct_helper.segment_roi(
            [gt, pred],
            mindim=[20 / spacing[0], 20 / spacing[1], -1],
        )
        rate = max([(zoom_roi[i].stop-zoom_roi[i].start)/gt.shape[i] for i in range(3)])
        zoom_roi = ct_helper.segment_roi(
            [gt, pred],
            mindim=[gt.shape[i]*rate for i in range(3)],
        )

        gt = gt[zoom_roi]
        pred = pred[zoom_roi]

    if ct is not None:
        ct = ct.copy()

        # ct = skimage.transform.downscale_local_mean(ct, scaledown)
        # gto = skimage.transform.downscale_local_mean(gto, scaledown)
        # gtmax = (gto.sum(axis=2).sum(axis=1).argmax(), gto.sum(axis=2).sum(axis=0).argmax(), gto.sum(axis=1).sum(axis=0).argmax())
        # transpose_idx = [0, 2, 1]
        # ct = ct.transpose(transpose_idx)
        # gto = gto.transpose(transpose_idx)
        # spacing2 = spacing[transpose_idx]
        ct_bounds = [0, ct.shape[0], 0, ct.shape[1], 0, ct.shape[2]]
        plot_ct = k3d.plot(grid=ct_bounds, name=ctlbl, grid_auto_fit=False, camera_auto_fit=0)
        subplots.add_plot(plot_ct, ctlbl)
        # .display()
        # ct[gtmax[0]:, gtmax[1]:, gtmax[2]:] = -1000
        # ct[gto > 0] = 1000.

        # plot_ct += k3d.volume(ct.astype(np.float32),
        #                       alpha_coef=100,
        #                       color_map=k3d.colormaps.matplotlib_color_maps.Turbo,
        #                       #   color_map=k3d.colormaps.paraview_color_maps.Nic_Edge,
        #                       #   color_map=k3d.colormaps.matplotlib_color_maps.gist_rainbow,
        #                       scaling=spacing,
        #                       color_range=[100, 1100],
        #                       bounds=ct_bounds,
        #                       #   gradient_step=1,
        #                       name='ct')
        origin = np.array([zoom_roi[i].start for i in range(3)])
        size = np.array([zoom_roi[i].stop-zoom_roi[i].start for i in range(3)])
        cube = k3d_tools.Cube(origin, size)
        # print('origin', [zoom_roi[i].start for i in range(3)])
        # print('size', [zoom_roi[i].stop-zoom_roi[i].start for i in range(3)])
        # print(cube.vertices)

        # plot_ct += k3d.mesh(cube.vertices, cube.indices, wireframe=True, scaling=spacing,)
        # plot_ct += k3d.points(cube.vertices, point_size=0.2, shader='3d', scaling=spacing, color=0x3f6bc5)

        # plot_ct += k3d.mip(ct.astype(np.float32),
        #                    color_map=k3d.colormaps.matplotlib_color_maps.Turbo,
        #                    #   color_map=k3d.colormaps.paraview_color_maps.Nic_Edge,
        #                    #   color_map=k3d.colormaps.matplotlib_color_maps.gist_rainbow,
        #                    scaling=spacing,
        #                    color_range=[-200, 900],
        #                    bounds=ct_bounds,
        #                    samples=100,
        #                    #   gradient_step=1,
        #                    name='ct')
        plot_ct += get_plot(ct, cmap('ct'), spacing, ct_bounds, 'ct-ct')
        plot_ct += get_plot(gto, cmap('gt'), spacing, ct_bounds, 'ct-gt')

        # k3d.mip(gto.astype(np.float32),
        #                    #   alpha_coef=100,
        #                    color_map=k3d.colormaps.matplotlib_color_maps.Greys,
        #                    scaling=spacing,
        #                    color_range=[0, 1],
        #                    bounds=ct_bounds,
        #                    samples=100,
        #                    #   gradient_step=1,
        #                    name='ct-gt')

        plot_ct.camera = [ct.shape[0]/2, ct.shape[1], -ct.shape[2], ct.shape[0]/3, ct.shape[1]/3, ct.shape[2]/3, 1, 0, 0]

    bounds = [0, gt.shape[0], 0, gt.shape[1], 0, gt.shape[2]]
    plot_gt = k3d.plot(grid=bounds, name=ctlbl, grid_auto_fit=False, camera_auto_fit=0)
    # plot_gt.display()
    # plot_gt.camera = camera
    subplots.add_plot(plot_gt, gtlbl)

    # plot_gt.camera = [gt.shape[0]/2, gt.shape[1], -gt.shape[2], gt.shape[0]/3, gt.shape[1]/3, gt.shape[2]/3, 1, 0, 0]
    plot_gt.colorbar_object_id = -1

    plot_pred = k3d.plot(grid=bounds, name=ctlbl, grid_auto_fit=False, camera_auto_fit=0)
    subplots.add_plot(plot_pred, 'Prediction')

    plot_pred.colorbar_object_id = -1
    tp = (gt > 0) & (pred > 0)
    fp = (~(gt > 0)) & (pred > 0)
    fn = ((gt > 0)) & ~(pred > 0)
    plot_gt += get_plot(tp, cmap('tp'), spacing, bounds, 'gt-tp')
    plot_gt += get_plot(fn, cmap('fn'), spacing, bounds, 'gt-fn')
    # plot_gt += k3d.mip(fn.astype(np.float32),
    #                    alpha_coef=100,
    #                    color_map=k3d.colormaps.matplotlib_color_maps.Reds,  # [0xFF3F00],
    #                    scaling=spacing,
    #                    color_range=[0, 1],
    #                    bounds=bounds,
    #                    show_legend=False,
    #                    gradient_step=1,
    #                    name='gt-fn')

    # plot_gt += k3d.mip(tp.astype(np.float32),
    #                    alpha_coef=100,
    #                    color_map=k3d.colormaps.matplotlib_color_maps.Greens,
    #                    #    color_map=[1.0000, 0.0000, 0.2667, 0.1059, 1.0000, 0.0000, 0.2667, 0.1059],
    #                    scaling=spacing,
    #                    color_range=[0, 2],
    #                    bounds=bounds,
    #                    gradient_step=1,
    #                    name='gt-tp')

    # v = k3d.voxels(tp.astype(np.float32),
    #                # alpha_coef=100,
    #                color_map=[0x82CD47],
    #                scaling=spacing,
    #                opacity=1,
    #                #    color_range=[0, 1],
    #                bounds=bounds,
    #                # translation=[0, 0, -gt.shape[0]],
    #                name='gt-tp2')
    # v.outlines = False
    # # plot_gt += v
    # v = k3d.voxels(fn.astype(np.float32),
    #                # alpha_coef=100,
    #                color_map=[0xFF3F00],
    #                scaling=spacing,
    #                opacity=1,
    #                #    color_range=[0, 1],
    #                bounds=bounds,
    #                # translation=[0, 0, -gt.shape[0]],
    #                name='gt-fn2')
    # v.outlines = False
    # # plot_gt += v

    # # plot_pred += k3d.volume(tp.astype(np.float32),
    # #                         alpha_coef=100,
    # #                         color_map=k3d.colormaps.matplotlib_color_maps.Greens,
    # #                         scaling=spacing,
    # #                         color_range=[0, 2],
    # #                         bounds=bounds, opacity=.4,

    # #                         gradient_step=1,
    # #                         # translation=[0, 0, -gt.shape[0]],
    # #                         name='pred-tp')
    # # plot_pred += k3d.volume(fp.astype(np.float32),
    # #                         alpha_coef=100,
    # #                         color_map=k3d.colormaps.basic_color_maps.Gold,
    # #                         scaling=spacing,
    # #                         color_range=[0, 1],
    # #                         bounds=bounds,
    # #                         show_legend=False,
    # #                         gradient_step=1,
    # #                         # translation=[0, 0, -gt.shape[0]],
    # #                         name='pred-fp')

    # v = k3d.voxels(tp.astype(np.float32),
    #                # alpha_coef=100,
    #                color_map=[0x379237],
    #                scaling=spacing,
    #                opacity=1,
    #                #    color_range=[0, 1],
    #                bounds=bounds,
    #                # translation=[0, 0, -gt.shape[0]],
    #                name='pred-tp2')
    # v.outlines = False
    # # plot_pred += v
    # v = k3d.voxels(fp.astype(np.float32),
    #                # alpha_coef=100,
    #                color_map=[0xF0FF42, 0xF0FF42],
    #                scaling=spacing,
    #                opacity=1,
    #                #    color_range=[0, 1],
    #                bounds=bounds,
    #                # translation=[0, 0, -gt.shape[0]],
    #                name='pred-fp2')
    # v.outlines = False
    # # plot_pred += v

    plot_pred += get_plot(tp, cmap('tp'), spacing, bounds, 'pred-tp')
    plot_pred += get_plot(fp, cmap('fp'), spacing, bounds, 'pred-fp')

    # subplots.sync_camera_view()
    return subplots


def plot_3d(ct, gt, pred, *, dst=None, spacing=None, ctlbl='CT', show_orig_size_ct=True,
            zoom2segments=True, show_ct=True, show_gt=True, show_pred=True, scaledown=(5, 5, 1), col=5, show=True, gtlbl="GroundTruth"):
    subplots = k3d_subplots()
    spacing = np.array([spacing[0], spacing[1], spacing[2]])
    # spacing = np.array(spacing)
    gt = skimage.transform.downscale_local_mean(gt, scaledown)
    pred = skimage.transform.downscale_local_mean(pred, scaledown)

    if show_ct and ct is not None:
        ct = skimage.transform.downscale_local_mean(ct, scaledown)

        if zoom2segments:
            zoom_roi = ct_helper.segment_roi(
                [gt, pred],
                mindim=[20 / spacing[0], 20 / spacing[1], -1],
            )
            rate = max([(zoom_roi[i].stop-zoom_roi[i].start)/gt.shape[i] for i in range(3)])
            zoom_roi = ct_helper.segment_roi(
                [gt, pred],
                mindim=[gt.shape[i]*rate for i in range(3)],
            )

            gt = gt[zoom_roi]
            pred = pred[zoom_roi]
            ct = ct[zoom_roi]

        plot_ct = k3d.plot(grid=get_bound(ct), name=ctlbl, grid_auto_fit=False, camera_auto_fit=0)
        subplots.add_plot(plot_ct, ctlbl, siz=1/(1+show_gt+show_pred))
        ct[ct > 400] = 0
        ct[gt > 0] = 1000
        plot_ct += get_plot(ct, spacing, 'ct', 'ct')
        plot_ct += get_plot(gt, spacing, 'gt', 'ct')
        plot_ct.camera = [ct.shape[0]/2, ct.shape[1], -ct.shape[2], ct.shape[0]/3, ct.shape[1]/3, ct.shape[2]/3, 1, 0, 0]

    tp = (gt > 0) & (pred > 0)
    fp = (~(gt > 0)) & (pred > 0)
    fn = (gt > 0) & ~(pred > 0)

    if show_gt:
        plot_gt = k3d.plot(grid=get_bound(gt), name=gtlbl, grid_auto_fit=False, camera_auto_fit=0)
        subplots.add_plot(plot_gt, gtlbl, siz=1/(1+show_ct+show_pred))
        plot_gt += get_plot(tp, spacing, 'tp', 'gt')
        plot_gt += get_plot(fn, spacing, 'fn', 'gt')

    # plot_gt.camera = [gt.shape[0]/2, gt.shape[1], -gt.shape[2], gt.shape[0]/3, gt.shape[1]/3, gt.shape[2]/3, 1, 0, 0]
    if show_pred:
        plot_pred = k3d.plot(grid=get_bound(pred), name=ctlbl, grid_auto_fit=False, camera_auto_fit=0)
        subplots.add_plot(plot_pred, 'Prediction', siz=1/(1+show_ct+show_gt))
        plot_pred += get_plot(tp, spacing, 'tp', 'pred')
        plot_pred += get_plot(fp, spacing, 'fp', 'pred')

    subplots.sync_camera_view()

    return subplots


def cmap(typ):
    if typ == 'fn':
        return k3d.colormaps.matplotlib_color_maps.Reds,
    if typ == 'tp':
        return k3d.colormaps.matplotlib_color_maps.Greens

    if typ == 'fp':
        return k3d.colormaps.basic_color_maps.Gold,

    if typ == 'ct':
        return k3d.colormaps.matplotlib_color_maps.Turbo,

    if typ == 'gt':
        return k3d.colormaps.matplotlib_color_maps.Greys,


def crange(typ):
    if typ in ['tp', 'fp']:
        return [0, 1]
    if typ in ['fn']:
        return [0, 1.3]
    if typ in ['gt']:
        return [0, 1.2]
    if typ == 'ct':
        return [0, 400]


def get_bound(arr):
    return [0, arr.shape[0], 0, arr.shape[1], 0, arr.shape[2]]


def get_plot(arr, spacing, typ, name):
    colormap = cmap(typ)
    color_range = crange(typ)
    # if typ == 'ct':
    # arr[arr > 300] = 0
    print(color_range)
    mode = 2
    if mode == 1:
        return k3d.mip(arr.astype(np.float32),
                       color_map=colormap,
                       color_range=color_range,
                       scaling=spacing,
                       samples=1024 if typ in ['ct', 'gt'] else 30,
                       bounds=get_bound(arr),
                       name=f'{name}-{typ}')

    return k3d.volume(arr.astype(np.float32),
                      color_map=colormap,
                      color_range=color_range,
                      scaling=spacing,
                      alpha_coef=100,
                      bounds=get_bound(arr), name=f'{name}-{typ}')
