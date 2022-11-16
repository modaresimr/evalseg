import copy
import hashlib
from collections import defaultdict
from enum import Enum

import cc3d
import numpy as np

from typing_extensions import Literal

from .. import common, geometry, ui
from ..common import Cache
from ..compress import compress_arr, decompress_arr
from . import MetricABS

epsilon = 0.0001

TP = "tp"
FP = "fp"
FN = "fn"
TN = "tn"
D = "D"
U = "U"
R = "R"
T = "T"
B = "B"
UI = "UI"

# pylint: disable-all
# pylint: disable=all
# pylint: skip-file


class MME(MetricABS):
    def __init__(self, debug={}):
        super().__init__(debug)

    def set_reference(self, reference: np.ndarray, spacing=None, **kwargs):
        super().set_reference(reference, spacing, **kwargs)
        spacing = self.spacing

        # pylint: disable=Need type annotation
        refc = reference
        gt_labels, gN = cc3d.connected_components(refc, return_N=True)
        gt_labels = gt_labels.astype(np.uint8)
        helperc = {}
        helperc["voxel_volume"] = spacing[0] * spacing[1] * spacing[2]
        helperc["gt_labels"] = gt_labels
        helperc["gN"] = gN

        gt_regions = geometry.expand_labels(gt_labels, spacing=spacing).astype(np.uint8)
        helperc["gt_regions"] = gt_regions
        helperc["components"] = {}
        helperc["total_gt_volume"] = 0
        for i in range(1, gN + 1):
            gt_component = gt_labels == i
            gt_region = gt_regions == i
            gt_border = geometry.find_binary_boundary(gt_component, mode="thick")
            gt_no_border = gt_component & ~gt_border
            gt_with_border = gt_component | gt_border
            in_dst = geometry.distance(gt_no_border, spacing=spacing, mode="in")
            out_dst = geometry.distance(gt_with_border, spacing=spacing, mode="out")
            gt_dst = out_dst + in_dst
            helperc["total_gt_volume"] += gt_component.sum() * helperc["voxel_volume"]

            skeleton = (geometry.skeletonize(gt_component, spacing=spacing) > 0)
            skeleton_dst = geometry.distance(skeleton, spacing=spacing, mode="out")
            # skeleton_dst[out_dst>skeleton_dst]=out_dst[out_dst>skeleton_dst]# this is an approximation so to avoid negative weights

            normalize_dst_inside = in_dst / (skeleton_dst + in_dst + epsilon)

            skel_dst = skeleton_dst - out_dst + in_dst

            deminutor = skeleton_dst - out_dst
            deminutor[np.abs(deminutor) < epsilon] = epsilon
            normalize_dst_outside = np.maximum(0, (out_dst) / deminutor)
            # normalize_dst_outside = normalize_dst_outside.clip(0, normalize_dst_outside.max())
            normalize_dst = normalize_dst_inside + normalize_dst_outside
            # idx = (119, 325)
            # tmp = {'skeleton_dst': skeleton_dst,
            #        'in_dst': in_dst, 'out_dst': out_dst, 'skel_dst': skel_dst,
            #        'normalize_dst_inside': normalize_dst_inside,
            #        'normalize_dst_outside': normalize_dst_outside,
            #        'normalize_dst': normalize_dst,
            #        }
            # print({k: tmp[k][idx] for k in tmp})
            helperc["components"][i] = compress_arr({
                "gt": gt_component,
                "gt_region": gt_region,
                "gt_border": gt_border,
                "skel_dst": skel_dst,
                "gt_dst": gt_dst,
                "gt_out_dst": out_dst,
                "gt_in_dst": in_dst,
                "gt_skeleton": skeleton,
                "gt_skeleton_dst": skeleton_dst,
                "skgt_normalized_dst": normalize_dst,
                "skgt_normalized_dst_in": normalize_dst_inside,
                "skgt_normalized_dst_out": normalize_dst_outside,
            })

            # if self.debug.get('show_precompute', 0):
            #     self.debug_helper(helperc['components'][i])

        self.helper = helperc

    def evaluate(self, test: np.ndarray, return_debug: bool = False, debug_prefix='', normalize_total_duration=True, **kwargs):
        calc_not_exist = False  # tmp
        reference = self.reference
        debug = self.debug
        assert (test.shape == reference.shape), "reference and test are not match"
        is2d = (test.ndim == 2 or test.shape[2] == 1)
        alpha1 = 0.001
        alpha2 = 1
        m_def = {d: {TP: 0, FP: 0, FN: 0, TN: 0} for d in [D, B, U, R, T]}

        dc = common.Object()
        info = {k: {} for k in m_def}

        dc.testc = test
        dc.helperc = self.helper

        resc = {"total": {}, "components": {}}

        dc.gt_labels, dc.gN = dc.helperc["gt_labels"], dc.helperc["gN"]
        dc.pred_labels, dc.pN = cc3d.connected_components(dc.testc, return_N=True)

        # extend gt_regions for not included components in prediction {
        dc.gt_regions = dc.helperc["gt_regions"]
        dc.gt_regions_idx = _get_component_idx(dc.gt_regions[dc.testc])
        dc.gt_region_mask = _get_merged_components(dc.gt_regions, dc.gt_regions_idx)
        dc.rel_gt_regions = np.zeros_like(dc.gt_regions)

        dc.rel_gt_regions[dc.gt_region_mask] = dc.gt_regions[dc.gt_region_mask]
        dc.gt_regions = geometry.expand_labels(dc.rel_gt_regions, spacing=self.spacing).astype(np.uint8)
        # }
        dc.rel = _get_rel_gt_pred(dc.gt_labels, dc.gN, dc.pred_labels, dc.pN, dc.gt_regions)
        # print('rel', dc.rel)
        resc["total"] = copy.deepcopy(m_def)
        dc.gts = {}

        dc.total_gt_volume = max(dc.helperc["total_gt_volume"], dc.helperc["voxel_volume"]) if normalize_total_duration else 1
        for ri in range(1, dc.gN + 1):
            dc.gts[ri] = dci = common.Object()
            hci = decompress_arr(dc.helperc["components"][ri])

            dci.component_gt = dc.rel["r+"][ri]["comp"]

            # dci.component_gt = hci['gt']
            m = copy.deepcopy(m_def)

            # dci.component_pred, dci.pred_comp_idx = _get_component_of(dc.pred_labels, dc.pred_labels[dci.component_gt], dc.pN)
            dci.component_pred = dc.rel["r+"][ri]["p+"]["merged_comp"]
            dci.pred_in_region = dc.rel["r+"][ri]["p+in_region"]["merged_comp"]

            if debug[UI] and is2d:
                ui_regions = dc.gt_regions.copy()
                ndst = hci["skgt_normalized_dst"].copy()
                ndst[ndst > 1] = 1
                # ndst[ndst > 2] = 0
                ndst[hci["gt_border"]] = 1

                ui_regions[ndst == 0] = 0
                ui.multi_plot_2d(
                    ndst,
                    dci.component_gt,
                    {
                        # 'all_pred': dci.component_pred,
                        # "region": ui_regions,
                        "pred_in_region": dci.pred_in_region,
                        **{f'p{i}': dc.rel['p+'][i]['comp']
                           for i in dc.rel["r+"][ri]["p+"]['idx']
                           }
                    },
                    z_titles=[f'{debug_prefix} ri={ri}'],
                    show_orig_size_ct=0,
                    # 'zoom2segments': 0
                )
            # dci.rel_p_gt_comps, dci.rel_p_gt_idx = _get_component_of(dc.gt_labels, dc.gt_labels[dci.component_pred], dc.gN)

            # Uniformity (TP and FN)....{
            # dci.tpuc = _Z(dc.rel, "r+", ri, "p+")
            # dci.tpuc = len(dc.rel["r+"][ri]["p+"]['idx'])
            # dci.tpu = 1 / dci.tpuc if dci.tpuc > 0 else 0

            # if calc_not_exist or dci.tpuc > 0:
            #     m[U][TP] = dci.tpu
            #     m[U][FN] = 1 - dci.tpu
            #     if debug[U]:
            #         print(f"  U tp+{f(dci.tpu)}  fn+{f(1-dci.tpu)}           Z[r+][{ri}][p+]=={f(dci.tpuc)}")
            #         if debug[UI] and is2d and False:
            #             if len(dc.rel["r+"][ri]["p+"]['idx']):
            #                 ui.multi_plot_img({
            #                     f'p{i}': dc.rel['p+'][i]['comp']
            #                     for i in dc.rel["r+"][ri]["p+"]['idx']
            #                 }, title=f"_Z={dci.tpuc} tpu={dci.tpu} ri={ri} pi={dc.rel['r+'][ri]['p+']['idx']}")
                # add_info(info, U, "r+", ri, dci.tpu, 1 - dci.tpu, 0)
            dci.tpuc = len(dc.rel["r+"][ri]["p+"]['idx'])
            if dci.tpuc > 0:
                dci.tpu = 1
                dci.fnu = dci.tpuc - 1
                dci.fpu = _Z(dc.rel, "r+", ri, "p+") - 1
                m[U][TP] += dci.tpu
                m[U][FN] += dci.fnu
                m[U][FP] += dci.fpu
                if debug[U]:
                    print(f"  U tp+{f(dci.tpu)}  fn+{f(dci.fnu)} fp+{f(dci.fpu)}      rel[r+][{ri}][p+]=={dci.tpuc}")
                add_info(info, U, "r+", ri, dci.tpu, dci.fnu, dci.fpu)
            # Uniformity}

            # dci.pred_in_region = dci.component_pred & (dc.gt_regions == ri)

            # # if a prediction contains two gt only consider the part related to gt
            # dc.gt_regions[dci.component_pred]
            # for l in dci.rel_gts:
            #     if l != ri:
            #         dci.pred_in_region = dci.pred_in_region & ~dc.gt_regions=

            # Total Volume================================={
            dci.tp_comp = dci.pred_in_region & dci.component_gt
            dci.fn_comp = (~dci.pred_in_region) & dci.component_gt
            dci.fp_comp = dci.pred_in_region & ~dci.component_gt

            dci.volume_gt = dci.component_gt.sum() * dc.helperc["voxel_volume"] / dc.total_gt_volume
            dci.volume_pred = (dci.pred_in_region.sum() * dc.helperc["voxel_volume"]) / dc.total_gt_volume

            dci.volume_tp = dci.tp_comp.sum() * dc.helperc["voxel_volume"] / dc.total_gt_volume
            dci.volume_fn = dci.fn_comp.sum() * dc.helperc["voxel_volume"] / dc.total_gt_volume  # dci.volume_gt - dci.volume_tp
            dci.volume_fp = dci.fp_comp.sum() * dc.helperc["voxel_volume"] / dc.total_gt_volume  # dci.volume_pred - dci.volume_tp

            m[T][TP] += dci.volume_tp
            m[T][FN] += dci.volume_fn
            m[T][FP] += dci.volume_fp
            if debug[T]:
                print(f" T tp+{f(dci.volume_tp)} fn+{f(dci.volume_fn)} fp+{f(dci.volume_fp)} rel[r+][{ri}] gtvol={f(dci.volume_gt)} pvol={f(dci.volume_pred)}")
            add_info(info, T, "r+", ri, dci.volume_tp, dci.volume_fn, dci.volume_fp,)
            # Total Volume}

            # Relative Volume=============================={
            dci.volume_tp_rate = dci.volume_tp / dci.volume_gt
            dci.volume_fn_rate = dci.volume_fn / dci.volume_gt
            dci.volume_fp_rate = min(1, dci.volume_fp / dci.volume_gt)
            if calc_not_exist or dci.volume_tp_rate > 0:
                m[R][TP] += dci.volume_tp_rate
                m[R][FN] += dci.volume_fn_rate
                m[R][FP] += min(1, dci.volume_fp_rate)

                if debug[R]:
                    print(f"    R tp+{f(dci.volume_tp_rate)} fn+{f(dci.volume_fn_rate)} fp+{f(dci.volume_fp_rate)}            rel[r+][{ri}])")

                add_info(info, R, "r+", ri, dci.volume_tp_rate, dci.volume_fn_rate, dci.volume_fp_rate,)
            # Relative Volume}

            # Boundary================================{
                dci.gt_skel = hci["gt_skeleton"]
                dci.skgtn_dst_in = hci["skgt_normalized_dst_in"]
                dci.skgtn_dst_out = hci["skgt_normalized_dst_out"]

                dci.boundary_tpc = dci.skgtn_dst_in[dci.tp_comp]
                dci.boundary_fnc = dci.skgtn_dst_in[dci.fn_comp]
                dci.boundary_fpc = dci.skgtn_dst_out[dci.fp_comp]

                if debug[B] or return_debug:
                    dci.boundary_tpc_v = np.zeros_like(dci.skgtn_dst_in)
                    dci.boundary_fpc_v = np.zeros_like(dci.skgtn_dst_in)
                    dci.boundary_fnc_v = np.zeros_like(dci.skgtn_dst_in)
                    dci.boundary_tpc_v[dci.tp_comp] = dci.boundary_tpc
                    dci.boundary_fpc_v[dci.fp_comp] = dci.boundary_fpc
                    dci.boundary_fnc_v[dci.fn_comp] = dci.boundary_fnc

                dci.boundary_gtc_sum = dci.boundary_tpc.sum()+dci.boundary_fnc.sum()
                if dci.boundary_gtc_sum > 0:
                    dci.boundary_fp = min(1, dci.boundary_fpc.sum()/dci.boundary_gtc_sum)
                    dci.boundary_fn = dci.boundary_fnc.sum() / dci.boundary_gtc_sum
                    dci.boundary_tp = dci.boundary_tpc.sum() / dci.boundary_gtc_sum

                    m[B][TP] += dci.boundary_tp
                    m[B][FN] += dci.boundary_fn
                    m[B][FP] += dci.boundary_fp
                    if debug[B]:
                        if debug[UI] and is2d:
                            tmp_out = dci.skgtn_dst_out.copy()
                            tmp_out[tmp_out > 2] = 0
                            ui.multi_plot_img({
                                "gtskel": dci.gt_skel,
                                "tp": dci.boundary_tpc_v,
                                "fn": dci.boundary_fnc_v,
                                "fp": dci.boundary_fpc_v,
                                'skeldst_in': dci.skgtn_dst_in,
                                'skeldst_out': tmp_out,
                                **{f'p+{i}': dc.rel['p+'][i]['comp'] for i in dc.rel["r+"][ri]["p+"]['idx']}
                            }, title=f'B tp+{f(dci.boundary_tp)} fn+{f(dci.boundary_fn)} fp+{f(dci.boundary_fp)}')

                        print(f"     B tp+{f(dci.boundary_tp)} fn+{f(dci.boundary_fn)} fp+{f(dci.boundary_fp)}  ri={ri}  ")
                    add_info(info, B, "r+", ri, dci.boundary_tp, dci.boundary_fn, dci.boundary_fp,)
            # } Boundary
            # old boundary {
            # # dci.border_gt = hci["gt_border"]
            # dci.gt_skel = hci["gt_skeleton"]
            # dci.border_pred_with_skel = geometry.find_binary_boundary(dci.pred_in_region, mode="inner")
            # dci.border_pred_with_skel |= dci.pred_in_region ^ dci.component_gt

            # # dci.border_pred_with_skel_inside_gt = dci.border_pred_with_skel & dci.component_gt
            # # dci.border_pred_with_skel_outside_gt = dci.border_pred_with_skel & (~dci.component_gt)

            # # dci.skgtn_dst_pred_in = dci.skgtn_dst_pred_in[dci.skgtn_dst_pred_in > 0]
            # # print('skgtn_dst_pred_in', dci.skgtn_dst_pred_in)
            # if return_debug or debug[B]:
            #     dci.skgtn_dst_pred_in_v = np.zeros(dci.skgtn_dst_in.shape)
            #     dci.skgtn_dst_pred_in_v[dci.border_pred_with_skel] = dci.skgtn_dst_in[dci.border_pred_with_skel]

            # dci.skgtn_dst_out = hci["skgt_normalized_dst_out"]
            # dci.skgtn_dst_pred_out = dci.skgtn_dst_out[dci.border_pred_with_skel]

            # if return_debug or debug[B]:
            #     dci.skgtn_dst_pred_out_v = np.zeros(dci.skgtn_dst_out.shape)
            #     dci.skgtn_dst_pred_out_v[dci.border_pred_with_skel] = dci.skgtn_dst_out[dci.border_pred_with_skel]

            # dci.skgtn_dst_pred = np.concatenate([dci.skgtn_dst_pred_out, dci.skgtn_dst_pred_in])
            # dci.skgtn_dst_pred = dci.skgtn_dst_pred[dci.skgtn_dst_pred > 0]

            # dci.boundary_fp = (min(1, dci.skgtn_dst_pred_out.mean())if len(dci.skgtn_dst_pred_out) > 0 else 0)
            # dci.boundary_fn = (dci.skgtn_dst_pred_in.mean()if len(dci.skgtn_dst_pred_in) > 0 else 0)
            # dci.boundary_tp = max(0, 1 - dci.boundary_fn)
            # if calc_not_exist or dci.volume_tp_rate > 0:
            #     m[B][TP] += dci.boundary_tp
            #     m[B][FN] += dci.boundary_fn
            #     m[B][FP] += dci.boundary_fp
            #     if debug[B]:
            #         # print(f"     B volume_tp_rate={dci.volume_tp_rate}")
            #         if debug[UI] and is2d:
            #             ui.multi_plot_img({
            #                 "gtskel": dci.gt_skel,
            #                 "pborder+minskel": dci.border_pred_with_skel,
            #                 'skeldst_in': dci.skgtn_dst_pred_in_v,
            #                 'skeldst_out': dci.skgtn_dst_pred_out_v,

            #                 **{f'p+{i}': dc.rel['p+'][i]['comp'] for i in dc.rel["r+"][ri]["p+"]['idx']}
            #             }, title=f'B tp+{f(dci.boundary_tp)} fn+{f(dci.boundary_fn)} fp+{f(dci.boundary_fp)}')

            #         print(f"     B tp+{f(dci.boundary_tp)} fn+{f(dci.boundary_fn)} fp+{f(dci.boundary_fp)}  ri={ri}  ")
            #     add_info(info, B, "r+", ri, dci.boundary_tp, dci.boundary_fn, dci.boundary_fp,)
            # OLD Boundary}

            # Detection===================================={
            tpd = int(dci.volume_tp_rate > alpha1)
            fnd = 1 - tpd
            fpd = dci.volume_fp_rate > alpha2
            m[D][TP] = tpd
            m[D][FN] = fnd
            m[D][FP] = fpd
            if debug["D"]:
                print(f" D TP+{f(tpd)}  FN+{f(fnd)} FP+{f(fpd)}     ri={ri}, p+={dc.rel['r+'][ri]['p+']['idx']} vtr={dci.volume_tp_rate}")
            add_info(info, D, "r+", ri, tpd, fnd, fpd)
            # Detection}

            # m[U][TP] += len(dci.pred_comp_idx) == 1
            # m[U][FN] += len(dci.pred_comp_idx) > 1

            for x in resc["total"]:
                for y in resc["total"][x]:
                    resc["total"][x][y] += m[x][y]

            resc["components"][ri] = {
                "MME": m,
                # 'hdn': self.info(dci.pred_dst / dci.max_dst_gt),
                # "skgtn": dci.skgtn_dst_pred.mean()if len(dci.skgtn_dst_pred)else 0,
                # "skgtn_tp": 1 - (np.clip(dci.skgtn_dst_pred, 0, 1).mean() if len(dci.skgtn_dst_pred) else 0),
                # "skgtn_fn": dci.skgtn_dst_pred_in.mean() if len(dci.skgtn_dst_pred_in) else 0,
                # "skgtn_fp": dci.skgtn_dst_pred_out.mean() if len(dci.skgtn_dst_pred_out) else 0,
            }
        resc["pN"] = dc.pN
        resc["gN"] = dc.gN

        # print(res)
        #     border_dst_shape=np.zeros(component_gt.shape,bool)
        #     border_dst_shape[border_pred]=dst[border_pred]
        #     plt.imshow(border_dst_shape)

        # print(dst[border_pred])

        # print(dst[0,0])
        dc.prs = {}
        for pi in range(1, dc.pN + 1):
            dc.prs[pi] = dci = common.Object()
            # dci.component_p = dc.pred_labels == pi
            dci.component_pred = dc.rel["p+"][pi]['comp']
            m = resc["total"]
            if debug[UI] and is2d:

                ui.multi_plot_2d(
                    None,
                    dci.component_pred,
                    {
                        # 'all_pred': dci.component_pred,
                        # "region": ui_regions,
                        **{f'r{i}': dc.rel['r+'][i]['comp']
                           for i in dc.rel["p+"][pi]["r+"]['idx']
                           }
                    },
                    gtlbl=f'pred {pi}',
                    z_titles=[f'{debug_prefix} pi={pi}'],
                    show_orig_size_ct=0,
                    # 'zoom2segments': 0
                )

            # DETECTION================================={

            # dci.rel_gts = dc.rel['p+'][pi]['r+']['idx']
            # dci.rel_gt_comps, dci.rel_gts = _get_component_of(gt_labels, gt_labels[dci.component_p], dc.gN)

            fpd = int(len(dc.rel["p+"][pi]["r+"]["idx"]) == 0)

            m[D][FP] += fpd
            if debug[D]:
                print(f" D FP+{f(fpd)}      pi={pi}, r={dc.rel['p+'][pi]['r+']['idx']}==0")
            add_info(info, D, "p+", pi, 0, 0, fpd)
            # DETECTION}
            # TOTAL DURATION================================={
            if fpd:
                dci.volume_fp = (dci.component_pred).sum() * dc.helperc["voxel_volume"] / dc.total_gt_volume
                m[T][FP] += dci.volume_fp
                if debug[T]:
                    print(f" T FP+{f(dci.volume_fp)}      pi={pi}, no related gt")
                add_info(info, T, "p+", pi, 0, 0, dci.volume_fp)
            # TOTAL DURATION}

            # resc['total'][U][FP] += len(dci.rel_gts) > 1

            # Uniformity FP=============================================={
            # fpuc = _Z(dc.rel, "p+", pi, "r+")
            # fpuc = len(dc.rel["p+"][pi]["r+"]['idx'])
            # if calc_not_exist or fpuc > 0:
            #     fpu = 1 - (1 / fpuc if fpuc > 0 else 0)
            #     resc["total"][U][FP] += fpu
            #     if debug[U]:
            #         print(f"  U fp+{f(fpu)}             Z[p+][{pi}][r+]=={f(fpuc)}")
            #     add_info(info, U, "p+", pi, 0, 0, fpu)
            # dci.fpuc = len(dc.rel["p+"][pi]["r+"]['idx'])
            # if dci.fpuc > 0:
            #     dci.fpu = dci.fpuc-1
            #     m[U][FP] += dci.fpu
            #     if debug[U]:
            #         print(f"  U fp+{f(dci.fpu)}      rel[p+][{pi}][r+]=={dci.fpuc}")
            #     add_info(info, U, "p+", pi, 0, 0, dci.fpu)
            # Uniformity}
        dc.resc = resc
        res = resc["total"]
        if return_debug:
            return res, dc
        return res

    # def info(self, na):
    #     return {
    #         'avg': na.mean() if len(na) > 0 else np.nan,
    #         'min': na.min() if len(na) > 0 else np.nan,
    #         '25': np.quantile(na, 0.25) if len(na) > 0 else np.nan,
    #         '50': np.quantile(na, 0.5) if len(na) > 0 else np.nan,
    #         '75': np.quantile(na, 0.75) if len(na) > 0 else np.nan,
    #         '95': np.quantile(na, 0.95) if len(na) > 0 else np.nan,
    #         'max': na.max() if len(na) > 0 else np.nan
    #     }


def _get_component_idx(classes):
    n = np.unique(classes)
    return n[n > 0]


def _Z(rel, X: Literal["r+", "p+"], xi: int, Y: Literal["r+", "p+"]):
    """For calculating related components in Uniformity
    @param X , Y: 'r+' or 'p+'
    @param xi : index of rel[X]

    the function _Z(rel, X, xi, Y), first selects components in Y name it as π that have
    intersection with X[xi] ; then, it finds components in X that are
    correlated with π, and return their quantity.

    For example, _Z(rel,'r+', ri, 'p+') first looks for predictions π that detect
    ri-th ground truth; it then returns the number of ground truth that are identified
    by those predictions (π).
    """
    assert X in ("r+", "p+"), f"Invalid X '{X}' should be 'r+' or 'p+'"
    assert Y in ("r+", "p+"), f"Invalid Y '{Y}' should be 'r+' or 'p+'"
    assert X != Y, f"Invalid X '{X}' should not be equal to Y {Y}"

    s = {}
    for yi in rel[X][xi][Y]["idx"]:
        for xi2 in rel[Y][yi][X]["idx"]:
            s[xi2] = 1

    return len(s)


# def _get_component_of(img, classes, max_component=None):
#     idx = np.s_[:]  #geometry.one_roi(img, return_index=True)
#     idx2 = np.s_[:]  #geometry.one_roi(classes, return_index=True)
#     max_component = max_component or classes.max()

#     pred_comp = [c for c in range(1, max_component + 1) if c in classes[idx2]]

#     component_pred = np.zeros(img.shape, bool)

#     for l in pred_comp:
#         component_pred[idx] |= img[idx] == l

#     return component_pred, pred_comp


def _get_merged_components(labels, classes):
    """get component with ids in classes and return a mask of all elements equal to classes

        @param labels: components contatining all labels
        @param classes: the classes to be filterd

    Returns
    -------
    np.array of bools
        The np array with the same size of labels and type of bool where labels[idx] in classes then out[idx]=True
    """
    component_merged = np.zeros(labels.shape, bool)

    for l in classes:
        component_merged |= labels == l

    return component_merged


def _get_rel_gt_pred(gt_labels, gN, pred_labels, pN, gt_regions):
    """calculate related components between prediction and ground truth
    it will also provide the prediction in valid region.

    """
    rel = {"r+": {}, "p+": {}}

    for ri in range(1, gN + 1):
        gt_comp = gt_labels == ri
        pidx = _get_component_idx(pred_labels[gt_comp])
        preds_in_region = np.zeros_like(pred_labels)
        region = gt_regions == ri
        preds_in_region[region] = pred_labels[region]
        rel["r+"][ri] = {
            "comp": gt_comp,
            "p+": {
                "idx": pidx,
                "merged_comp": _get_merged_components(pred_labels, pidx),
            },
            "p+in_region": {
                "idx": pidx,
                "labels": preds_in_region,
                "merged_comp": _get_merged_components(preds_in_region, pidx),
            },
        }
    for pi in range(1, pN + 1):
        pred_comp = pred_labels == pi
        ridx = _get_component_idx(gt_labels[pred_comp])
        rel["p+"][pi] = {
            "comp": pred_comp,
            "r+": {
                "idx": ridx,
                "merged_comp": _get_merged_components(gt_labels, ridx),
            },
        }

    return rel


def f(n):

    return np.round(n * 1.0, 2)


def add_info(info, property, p_or_r, indx, tp, fn, fp):
    if not (property in info):
        info[property] = {}
    if not (p_or_r in info[property]):
        info[property][p_or_r] = {}
    if not (indx in info[property][p_or_r]):
        info[property][p_or_r][indx] = {"tp": 0, "fp": 0, "fn": 0}
    info[property][p_or_r][indx]["tp"] += tp
    info[property][p_or_r][indx]["fp"] += fp
    info[property][p_or_r][indx]["fn"] += fn
