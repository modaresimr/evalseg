import copy
import hashlib
from collections import defaultdict
from enum import Enum
import psutil
import cc3d
import numpy as np
import gc
from typing_extensions import Literal

from .. import common, geometry, ui
from ..common import Cache
from ..io import SegmentArray
from ..compress import compress_arr, decompress_arr
from . import MetricABS
import auto_profiler
epsilon = 0.001

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
        self.optimize_memory = True

    def _calc_component_info(self, component: SegmentArray, cid):
        # print('cid=', cid)
        # return {}
        spacing = component.voxelsize
        gt_component = component.todense()
        gt_border = geometry.find_binary_boundary(gt_component, mode="thick")
        gt_no_border = gt_component & ~gt_border
        gt_with_border = gt_component | gt_border
        gt_border_seg = SegmentArray(gt_border, multi_part=False)
        safe_roi = geometry.calc_safe_roi(gt_border_seg.shape, gt_border_seg.roi, roi_rate=3)
        segment_roi = gt_border_seg.roi

        in_dst = geometry.distance(gt_no_border, spacing=spacing, mode="in", mask_roi=segment_roi)

        out_dst = geometry.distance(gt_with_border, spacing=spacing, mask_roi=safe_roi, mode="out")

        if self.optimize_memory:
            del gt_with_border
            del gt_no_border
            del gt_border

        # gt_dst = out_dst + in_dst

        skeleton = (geometry.skeletonize(gt_component, spacing=spacing, mask_roi=segment_roi) > 0)
        if self.optimize_memory:
            del gt_component
            gc.collect()
        skeleton_dst = geometry.distance(skeleton, spacing=spacing, mask_roi=safe_roi, mode="out")
        # skeleton_dst[out_dst>skeleton_dst]=out_dst[out_dst>skeleton_dst]# this is an approximation so to avoid negative weights
        skeleton = SegmentArray(skeleton, mask_roi=segment_roi, multi_part=False)

        deminutor = skeleton_dst + in_dst
        deminutor[np.abs(deminutor) < 1] = 1
        normalize_dst_inside = in_dst / deminutor
        normalize_dst_inside_seg = SegmentArray(normalize_dst_inside, mask_roi=gt_border_seg.roi, multi_part=False)

        if self.optimize_memory:
            del deminutor
            del in_dst
            del normalize_dst_inside
            gc.collect()

        deminutor = skeleton_dst - out_dst
        deminutor[np.abs(deminutor) < 1] = 1
        normalize_dst_outside = out_dst / deminutor
        normalize_dst_outside_seg = SegmentArray(np.clip(normalize_dst_outside, 0, 5), fill_value=5, mask_roi=safe_roi, multi_part=False)
        if self.optimize_memory:
            del deminutor
            del out_dst
            del skeleton_dst
            del normalize_dst_outside
            gc.collect()

        # normalize_dst_outside = normalize_dst_outside.clip(0, normalize_dst_outside.max())
        #

        res = {
            "gt_border": gt_border_seg,  # for visu
            "gt_skeleton": skeleton,  # for visu
            "skgt_normalized_dst_in": normalize_dst_inside_seg,
            "skgt_normalized_dst_out": normalize_dst_outside_seg,
        }

        if not self.optimize_memory:
            res['gt'] = SegmentArray(gt_component)
            # res['gt_region'] = SegmentArray(gt_regions == i)
            skel_dst = skeleton_dst - out_dst + in_dst
            res["skel_dst"] = SegmentArray(skel_dst, fill_value=skel_dst.max())
            res["gt_skeleton_dst"] = SegmentArray(skeleton_dst, fill_value=skeleton_dst.max())

            res["gt_out_dst"] = SegmentArray(out_dst, fill_value=out_dst.max())
            res["gt_in_dst"] = SegmentArray(in_dst)
            gt_dst = in_dst+out_dst
            res["gt_dst"] = SegmentArray(gt_dst, fill_value=gt_dst.max())
            normalize_dst = normalize_dst_inside_seg + normalize_dst_outside_seg
            res["skgt_normalized_dst"] = SegmentArray(normalize_dst, fill_value=normalize_dst.max())
        return res

    def set_reference(self, reference: SegmentArray, **kwargs):
        # def set_reference(self, reference: np.ndarray, spacing=None, **kwargs):
        super().set_reference(reference, **kwargs)
        if hasattr(self, 'helper'):
            del self.helper
        assert isinstance(reference, SegmentArray)
        assert reference.dtype == bool

        spacing = self.reference.voxelsize

        # pylint: disable=Need type annotation

        gt_labels, gN = get_components_from_segment(reference)
        # refc = reference.todense()
        # gt_labels, gN =geometry.connected_components(refc, return_N=True)
        # gt_labels = gt_labels.astype(np.uint8)

        helperc = {}
        helperc["voxel_volume"] = spacing[0] * spacing[1] * spacing[2]
        helperc["gt_labels"] = gt_labels
        helperc["gN"] = gN

        gt_regions = geometry.expand_labels(gt_labels.todense(), spacing=spacing).astype(np.uint8)
        helperc["gt_regions"] = SegmentArray(gt_regions)

        helperc["total_gt_volume"] = reference.sum()*helperc["voxel_volume"]

        items = [{'cid': i, 'component': gt_labels == i} for i in range(1, gN + 1)]
        # items = [{'cid': i, 'component': None} for i in range(1, gN + 1)]

        maxcpu = psutil.virtual_memory().available//(30 * 1024 * 1024 * 1024)+1
        res = common.parallel_runner(self._calc_component_info, items, parallel=0, max_cpu=maxcpu, maxtasksperchild=1)
        # res = common.parallel_runner(self.__calc_component_info, items, parallel=0, max_cpu=1, maxtasksperchild=1)

        helperc["components"] = {k['cid']: v for k, v in res}
        self.helper = helperc

    # @auto_profiler.Profiler(filterExternalLibraries=False, depth=20)
    def evaluate(self, test: SegmentArray, return_debug: bool = False, debug_prefix='', normalize_total_duration=True, **kwargs):
        assert isinstance(test, SegmentArray)
        assert test.dtype == bool
        calc_not_exist = False  # tmp
        reference = self.reference
        debug = self.debug
        assert (test.shape == reference.shape), "reference and test are not match"
        is2d = (len(test.shape) == 2 or test.shape[2] == 1)
        alpha1 = 0.001
        alpha2 = 1
        m_def = {d: {TP: 0, FN: 0, FP: 0, TN: 0} for d in [D, B, U, R, T]}

        # dc = common.Object()
        info = {k: {} for k in m_def}

        # testc = test.todense()
        helperc = self.helper

        resc = {"total": {}, "components": {}}

        gt_labels, gN = helperc["gt_labels"], helperc["gN"]

        # dc.gt_labels = dc.helperc["gt_labels"].segments
        # dc.gN = len(dc.helperc["gt_labels"].segments)

        # dc.pred_labels, dc.pN = geometry.connected_components(dc.testc, return_N=True)
        pred_labels, pN = get_components_from_segment(test)
        # dc.pN = len(dc.testc.segments)

        # extend gt_regions for not included components in prediction {

        gt_regions = get_gt_regions_for_pred(helperc["gt_regions"], test)

        # rel_gt_regions = np.zeros(gt_regions.shape, gt_regions.dtype)
        # rel_gt_regions[gt_region_mask] = gt_regions[gt_region_mask]
        # gt_regions = geometry.expand_labels(rel_gt_regions, spacing=reference.voxelsize).astype(np.uint8)

        # }
        rel = _get_rel_gt_pred(gt_labels, gN, pred_labels, pN, gt_regions)
        # print('rel', dc.rel)
        resc["total"] = copy.deepcopy(m_def)
        # gts = {}

        total_gt_volume = max(helperc["total_gt_volume"], helperc["voxel_volume"]) if normalize_total_duration else 1
        for ri in range(1, gN + 1):
            # gts[ri] = dci = common.Object()
            hci = helperc["components"][ri]

            component_gt = rel["r+"][ri]["comp"]

            # dci.component_gt = hci['gt']
            m = copy.deepcopy(m_def)

            # dci.component_pred, dci.pred_comp_idx = _get_component_of(pred_labels, pred_labels[dci.component_gt], pN)
            component_pred = rel["r+"][ri]["p+"]["merged_comp"]
            pred_in_region = rel["r+"][ri]["p+in_region"]["merged_comp"]

            if debug[UI] and is2d:
                ui_regions = gt_regions.todense()
                ndst_in = hci["skgt_normalized_dst_in"]
                ndst_out = hci["skgt_normalized_dst_out"]
                ndst = (ndst_in+ndst_out).todense()
                ndst[ndst > 1] = 1
                # ndst[ndst > 2] = 0
                ndst[hci["gt_border"].todense()] = 1

                ui_regions[ndst == 0] = 0
                ui.multi_plot_2d(
                    ndst,
                    component_gt.todense(),
                    {
                        # 'all_pred': dci.component_pred,
                        # "region": ui_regions,
                        "pred_in_region": pred_in_region.todense(),
                        **{f'p{i}': rel['p+'][i]['comp'].todense()
                           for i in rel["r+"][ri]["p+"]['idx']
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
            tpuc = len(rel["r+"][ri]["p+"]['idx'])
            if tpuc > 0:
                tpu = 1
                fnu = tpuc - 1
                fpu = _Z(rel, "r+", ri, "p+") - 1
                m[U][TP] += tpu
                m[U][FN] += fnu
                m[U][FP] += fpu
                if debug[U]:
                    print(f"  U tp+{f(tpu)}  fn+{f(fnu)} fp+{f(fpu)}      rel[r+][{ri}][p+]=={tpuc}")
                add_info(info, U, "r+", ri, tpu, fnu, fpu)
            # Uniformity}

            # dci.pred_in_region = dci.component_pred & (gt_regions == ri)

            # # if a prediction contains two gt only consider the part related to gt
            # dc.gt_regions[dci.component_pred]
            # for l in dci.rel_gts:
            #     if l != ri:
            #         dci.pred_in_region = dci.pred_in_region & ~dc.gt_regions=

            # Total Volume================================={
            tp_comp = pred_in_region & component_gt
            fn_comp = (~pred_in_region) & component_gt
            fp_comp = pred_in_region & ~component_gt

            volume_gt = component_gt.sum() * helperc["voxel_volume"] / total_gt_volume
            volume_pred = (pred_in_region.sum() * helperc["voxel_volume"]) / total_gt_volume

            volume_tp = tp_comp.sum() * helperc["voxel_volume"] / total_gt_volume
            volume_fn = fn_comp.sum() * helperc["voxel_volume"] / total_gt_volume  # volume_gt - volume_tp
            volume_fp = fp_comp.sum() * helperc["voxel_volume"] / total_gt_volume  # volume_pred - volume_tp

            m[T][TP] += volume_tp
            m[T][FN] += volume_fn
            m[T][FP] += volume_fp
            if debug[T]:
                print(f" T tp+{f(volume_tp)} fn+{f(volume_fn)} fp+{f(volume_fp)} rel[r+][{ri}] gtvol={f(volume_gt)} pvol={f(volume_pred)}")
            add_info(info, T, "r+", ri, volume_tp, volume_fn, volume_fp,)
            # Total Volume}

            # Relative Volume=============================={
            volume_tp_rate = volume_tp / volume_gt
            volume_fn_rate = volume_fn / volume_gt
            volume_fp_rate = min(1, volume_fp / volume_gt)
            if calc_not_exist or volume_tp_rate > 0:
                m[R][TP] += volume_tp_rate
                m[R][FN] += volume_fn_rate
                m[R][FP] += min(1, volume_fp_rate)

                if debug[R]:
                    print(f"    R tp+{f(volume_tp_rate)} fn+{f(volume_fn_rate)} fp+{f(volume_fp_rate)}            rel[r+][{ri}])")

                add_info(info, R, "r+", ri, volume_tp_rate, volume_fn_rate, volume_fp_rate,)
            # Relative Volume}

            # Boundary================================{
                gt_skel = hci["gt_skeleton"]  # .todense()
                skgtn_dst_in = hci["skgt_normalized_dst_in"]  # .todense()
                skgtn_dst_out = hci["skgt_normalized_dst_out"]  # .todense()

                boundary_tpc = skgtn_dst_in[tp_comp]
                boundary_fnc = skgtn_dst_in[fn_comp]
                boundary_fpc = skgtn_dst_out[fp_comp]

                if debug[B] or return_debug:
                    boundary_tpc_v = np.zeros(skgtn_dst_in.shape, skgtn_dst_in.dtype)
                    boundary_fpc_v = np.zeros(skgtn_dst_in.shape, skgtn_dst_in.dtype)
                    boundary_fnc_v = np.zeros(skgtn_dst_in.shape, skgtn_dst_in.dtype)
                    boundary_tpc_v[tp_comp.todense()] = boundary_tpc
                    boundary_fpc_v[fp_comp.todense()] = boundary_fpc
                    boundary_fnc_v[fn_comp.todense()] = boundary_fnc

                boundary_fnc_sum = boundary_fnc.sum(dtype=np.float64)
                boundary_tpc_sum = boundary_tpc.sum(dtype=np.float64)
                boundary_gtc_sum = boundary_tpc_sum+boundary_fnc_sum
                if boundary_gtc_sum > 0:
                    boundary_fp = min(1, boundary_fpc.sum(dtype=np.float64)/boundary_gtc_sum)
                    boundary_fn = boundary_fnc_sum / boundary_gtc_sum
                    boundary_tp = boundary_tpc_sum / boundary_gtc_sum

                    m[B][TP] += boundary_tp
                    m[B][FN] += boundary_fn
                    m[B][FP] += boundary_fp
                    if debug[B]:
                        if debug[UI] and is2d:
                            tmp_out = skgtn_dst_out.todense()
                            tmp_out[tmp_out > 2] = 0
                            ui.multi_plot_img({
                                "gtskel": gt_skel.todense(),
                                "tp": boundary_tpc_v,
                                "fn": boundary_fnc_v,
                                "fp": boundary_fpc_v,
                                'skeldst_in': skgtn_dst_in.todense(),
                                'skeldst_out': tmp_out,
                                **{f'p+{i}': rel['p+'][i]['comp'].todense() for i in rel["r+"][ri]["p+"]['idx']}
                            }, title=f'B tp+{f(boundary_tp)} fn+{f(boundary_fn)} fp+{f(boundary_fp)}')

                        print(f"     B tp+{f(boundary_tp)} fn+{f(boundary_fn)} fp+{f(boundary_fp)}  ri={ri}  ")
                    add_info(info, B, "r+", ri, boundary_tp, boundary_fn, boundary_fp,)
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
            tpd = int(volume_tp_rate > alpha1)
            fnd = 1 - tpd
            fpd = volume_fp_rate > alpha2
            m[D][TP] = tpd
            m[D][FN] = fnd
            m[D][FP] = fpd
            if debug["D"]:
                print(f" D TP+{f(tpd)}  FN+{f(fnd)} FP+{f(fpd)}     ri={ri}, p+={rel['r+'][ri]['p+']['idx']} vtr={volume_tp_rate}")
            add_info(info, D, "r+", ri, tpd, fnd, fpd)
            # Detection}

            # m[U][TP] += len(pred_comp_idx) == 1
            # m[U][FN] += len(pred_comp_idx) > 1

            for x in resc["total"]:
                for y in resc["total"][x]:
                    resc["total"][x][y] += m[x][y]

            resc["components"][ri] = {
                "MME": m,
                # 'hdn': self.info(pred_dst / max_dst_gt),
                # "skgtn": skgtn_dst_pred.mean()if len(skgtn_dst_pred)else 0,
                # "skgtn_tp": 1 - (np.clip(dci.skgtn_dst_pred, 0, 1).mean() if len(dci.skgtn_dst_pred) else 0),
                # "skgtn_fn": dci.skgtn_dst_pred_in.mean() if len(dci.skgtn_dst_pred_in) else 0,
                # "skgtn_fp": dci.skgtn_dst_pred_out.mean() if len(dci.skgtn_dst_pred_out) else 0,
            }
        resc["pN"] = pN
        resc["gN"] = gN

        # print(res)
        #     border_dst_shape=np.zeros(component_gt.shape,bool)
        #     border_dst_shape[border_pred]=dst[border_pred]
        #     plt.imshow(border_dst_shape)

        # print(dst[border_pred])

        # print(dst[0,0])
        # prs = {}
        for pi in range(1, pN + 1):
            # prs[pi] = dci = common.Object()
            # dci.component_p = pred_labels == pi
            component_pred = rel["p+"][pi]['comp']
            m = resc["total"]
            if debug[UI] and is2d:

                ui.multi_plot_2d(
                    None,
                    component_pred.todense(),
                    {
                        # 'all_pred': dci.component_pred,
                        # "region": ui_regions,
                        **{f'r{i}': rel['r+'][i]['comp'].todense()
                           for i in rel["p+"][pi]["r+"]['idx']
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

            fpd = int(len(rel["p+"][pi]["r+"]["idx"]) == 0)

            m[D][FP] += fpd
            if debug[D]:
                print(f" D FP+{f(fpd)}      pi={pi}, r={rel['p+'][pi]['r+']['idx']}==0")
            add_info(info, D, "p+", pi, 0, 0, fpd)
            # DETECTION}
            # TOTAL DURATION================================={
            if fpd:
                volume_fp = (component_pred).sum() * helperc["voxel_volume"] / total_gt_volume
                m[T][FP] += volume_fp
                if debug[T]:
                    print(f" T FP+{f(volume_fp)}      pi={pi}, no related gt")
                add_info(info, T, "p+", pi, 0, 0, volume_fp)
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
        res = resc["total"]
        if return_debug:
            return res, None
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


def _get_merged_components(labels: SegmentArray, classes):
    """get component with ids in classes and return a mask of all elements equal to classes

        @param labels: components contatining all labels
        @param classes: the classes to be filterd

    Returns
    -------
    np.array of bools
        The np array with the same size of labels and type of bool where labels[idx] in classes then out[idx]=True
    """
    # component_merged = np.zeros(labels.shape, bool)
    component_merged = SegmentArray(np.zeros((0, 0, 0), bool), shape=labels.shape, spoint=[0, 0, 0], dtype=bool)

    for l in classes:
        component_merged = component_merged.__or__(labels.__eq__(l, calc_roi=False, multi_part=False), multi_part=False)

    return component_merged


def _get_rel_gt_pred(gt_labels: SegmentArray, gN, pred_labels: SegmentArray, pN, gt_regions: np.ndarray):
    """calculate related components between prediction and ground truth
    it will also provide the prediction in valid region.

    """
    rel = {"r+": {}, "p+": {}}
    for ri in range(1, gN + 1):
        gt_comp = gt_labels == ri
        pidx = _get_component_idx(pred_labels[gt_comp])

        region = gt_regions == ri
        preds_in_region = np.zeros(pred_labels.shape, pred_labels.dtype)
        # preds_in_region[region] = region&pred_labels[region]
        # preds_in_region = SegmentArray(preds_in_region, multi_part=False)
        x = region[region.roi]
        preds_in_region[region.roi][x] = pred_labels[region.roi][x]
        preds_in_region = SegmentArray(preds_in_region, multi_part=False)
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


def get_components_from_segment(segment: SegmentArray):
    # res = np.zeros(segment.shape, np.uint8)
    rescc, n = geometry.connected_components(segment[segment.roi], return_N=True)

    spoint = [r.start for r in segment.roi]
    res = SegmentArray(rescc, segment.voxelsize, dtype=np.uint8, shape=segment.shape, spoint=spoint)

    return res, n


def get_gt_regions_for_pred(orig_gt_regions: SegmentArray, test: SegmentArray):
    intestgt = orig_gt_regions[test.roi].copy()
    # intestgt[test.todense()] = orig_gt_regions[test]

    gt_regions_idx = _get_component_idx(intestgt[test[test.roi]])
    # gt_region_mask_old = _get_merged_components(orig_gt_regions, gt_regions_idx).todense()

    # rel_gt_regions = np.zeros(orig_gt_regions.shape, orig_gt_regions.dtype)
    # rel_gt_regions[gt_region_mask_old] = orig_gt_regions[gt_region_mask_old]
    # gt_regions_old = geometry.expand_labels(rel_gt_regions, spacing=test.voxelsize).astype(np.uint8)

    intestgt[~np.isin(intestgt, gt_regions_idx)] = 0

    # if not np.all(rel_gt_regions == gt_regions_2):
    #     print('error')
    intestgt = geometry.expand_labels(intestgt, spacing=test.voxelsize).astype(np.uint8)
    gt_regions = SegmentArray(intestgt, shape=test.shape, spoint=[r.start for r in test.roi],
                              calc_roi=False, multi_part=False, fill_value=0)

    # ui.multi_plot_img({'orig': orig_gt_regions.todense(), 'regions': gt_regions.todense(), 'test': test.todense()}, interactive=True)

    return gt_regions


def get_gt_regions_for_pred2(orig_gt_regions: SegmentArray, test: SegmentArray):
    intestgt = np.zeros([r.stop-r.start for r in test.roi], orig_gt_regions.dtype)
    rel_gt = orig_gt_regions[test]

    gt_regions_idx = _get_component_idx(rel_gt)
    # gt_region_mask_old = _get_merged_components(orig_gt_regions, gt_regions_idx).todense()

    # rel_gt_regions = np.zeros(orig_gt_regions.shape, orig_gt_regions.dtype)
    # rel_gt_regions[gt_region_mask_old] = orig_gt_regions[gt_region_mask_old]
    # gt_regions_old = geometry.expand_labels(rel_gt_regions, spacing=test.voxelsize).astype(np.uint8)

    gt_region_mask = np.isin(orig_gt_regions.todense(), gt_regions_idx)
    gt_region_mask = SegmentArray(gt_region_mask)

    rroi = gt_region_mask.roi
    mask_in_roi = gt_region_mask[gt_region_mask.roi]
    gt_regions_2 = np.zeros(mask_in_roi.shape, orig_gt_regions.dtype)
    gt_regions_2[mask_in_roi] = orig_gt_regions[gt_region_mask]
    # if not np.all(rel_gt_regions == gt_regions_2):
    #     print('error')
    gt_regions_2 = geometry.expand_labels(gt_regions_2, spacing=test.voxelsize).astype(np.uint8)
    gt_regions = SegmentArray(gt_regions_2, shape=test.shape, spoint=[r.start for r in rroi],
                              calc_roi=False, multi_part=False, fill_value=gt_regions_idx[0])

    # ui.multi_plot_img({'orig': orig_gt_regions.todense(), 'regions': gt_regions.todense(), 'test': test.todense()}, interactive=True)

    return gt_regions
