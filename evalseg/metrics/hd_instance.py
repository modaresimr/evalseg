import copy
from enum import Enum
from typing_extensions import Literal

import cc3d
import numpy as np
import skimage

from .. import common, geometry
from ..common import Cache
from . import MetricABS, mme

epsilon = 0.00001

# pylint: disable-all
# pylint: disable=all
# pylint: skip-file


class hd_instance(MetricABS):
    def __init__(self, num_classes, debug={}):
        super().__init__(num_classes, debug)
        pass

    def calculate_info(
        cls,
        reference: np.ndarray,
        spacing: np.ndarray = None,
        num_classes: int = 2,
        **kwargs
    ):
        helper = {}

        helper["voxel_volume"] = spacing[0] * spacing[1] * spacing[2]
        helper["class"] = {}
        for c in range(num_classes):
            # print(f'class={c}')

            refc = reference == c

            gt_labels, gN = cc3d.connected_components(refc, return_N=True)
            gt_labels = gt_labels.astype(np.uint8)
            helperc = {}
            helper["class"][c] = helperc
            helperc["gt_labels"] = gt_labels
            helperc["gN"] = gN

            gt_regions = geometry.expand_labels(
                gt_labels, spacing=spacing
            ).astype(np.uint8)
            helperc["gt_regions"] = gt_regions
            helperc["components"] = {}
            for i in range(1, gN + 1):
                gt_component = gt_labels == i
                gt_region = gt_regions == i
                gt_border = geometry.find_binary_boundary(
                    gt_component, mode="thick"
                )
                gt_no_border = gt_component & ~gt_border
                gt_with_border = gt_component | gt_border
                in_dst = geometry.distance(
                    gt_no_border, spacing=spacing, mode="in"
                )
                out_dst = geometry.distance(
                    gt_with_border, spacing=spacing, mode="out"
                )
                gt_dst = out_dst - in_dst

                helperc["components"][i] = {
                    "gt": gt_component,
                    "gt_region": gt_region,
                    "gt_border": gt_border,
                    "gt_dst": gt_dst,
                    "gt_out_dst": out_dst,
                    "gt_in_dst": in_dst,
                }

        return helper

    def evaluate(
        self, test: np.ndarray, return_debug_data: bool = False, **kwargs
    ):
        calc_not_exist = False  # tmp
        reference = self.reference
        helper = self.helper
        debug = self.debug
        assert (
            test.shape == reference.shape
        ), "reference and test are not match"

        alpha1 = 0.1
        alpha2 = 1

        res = {"class": {}}

        data = {}
        m_def = {"HD_max": 0, "HD95": 0, "HD_avg": 0}
        for c in range(self.num_classes):
            data[c] = dc = common.Object()

            dc.testc = test == c
            dc.helperc = helper["class"][c]

            res["class"][c] = resc = {"total": {}, "components": {}}

            dc.gt_labels, dc.gN = dc.helperc["gt_labels"], dc.helperc["gN"]
            dc.pred_labels, dc.pN = cc3d.connected_components(
                dc.testc, return_N=True
            )

            # extend gt_regions for not included components in prediction
            dc.gt_regions = dc.helperc["gt_regions"]
            dc.gt_regions_idx = dc.gt_regions[dc.testc].unique()
            dc.rel_gt_regions = dc.gt_regions[
                mme._get_merged_components(dc.gt_regions, dc.gt_regions_idx)
            ]
            dc.gt_regions = geometry.expand_labels(
                dc.rel_gt_regions, spacing=self.spacing
            ).astype(np.uint8)

            dc.rel = mme._get_rel_gt_pred(
                dc.gt_labels, dc.gN, dc.pred_labels, dc.pN, dc.gt_regions
            )

            dc.gts = {}
            for ri in range(1, dc.gN + 1):
                dc.gts[ri] = dci = common.Object()
                hci = dc.helperc["components"][ri]

                dci.component_gt = dci.component_gt = hci["gt"]
                m = copy.deepcopy(m_def)

                # dci.component_pred, dci.pred_comp_idx = _get_component_of(dc.pred_labels, dc.pred_labels[dci.component_gt], dc.pN)
                dci.component_pred = dc.rel["r+"][ri]["p+"]["merged_comp"]

                # dci.rel_p_gt_comps, dci.rel_p_gt_idx = _get_component_of(dc.gt_labels, dc.gt_labels[dci.component_pred], dc.gN)

                # dci.pred_in_region = dci.component_pred & (dc.gt_regions == ri)
                dci.pred_in_region = dc.rel["r+"][ri]["p+in_region"]["comp"]
                # # if a prediction contains two gt only consider the part related to gt
                # dc.gt_regions[dci.component_pred]
                # for l in dci.rel_gts:
                #     if l != ri:
                #         dci.pred_in_region = dci.pred_in_region & ~dc.gt_regions=

                dci.border_gt = hci["gt_border"]
                dci.border_pred = geometry.find_binary_boundary(
                    dci.pred_in_region, mode="inner"
                )

                # calculate HD distance=========================={
                dci.dst_border_gt2pred = hci["gt_dst"][dci.border_pred]
                dci.dst_border_gt2pred_abs = np.abs(dci.dst_border_gt2pred)
                # dci.dst_border_gt2pred_v = np.zeros(dci.border_pred.shape)
                # dci.dst_border_gt2pred_v[dci.border_pred] = hci['gt_dst'][dci.border_pred]

                dci.gt_hd = (
                    dci.dst_border_gt2pred_abs.max()
                    if len(dci.dst_border_gt2pred) > 0
                    else np.nan
                )
                dci.gt_hd_avg = (
                    dci.dst_border_gt2pred_abs.mean()
                    if len(dci.dst_border_gt2pred) > 0
                    else np.nan
                )
                dci.gt_hd95 = (
                    np.quantile(dci.dst_border_gt2pred_abs, 0.95)
                    if len(dci.dst_border_gt2pred) > 0
                    else np.nan
                )

                dci.pred_border_dst = geometry.distance(
                    dci.component_pred,
                    mode="both",
                    mask=dci.rel_gt_comps | dci.component_pred,
                    spacing=self.spacing,
                )
                dci.dst_border_pred2gt = dci.pred_border_dst[dci.border_gt]
                dci.dst_border_pred2gt_abs = np.abs(dci.dst_border_pred2gt)

                # dci.dst_border_pred2gt_v = np.zeros(dci.border_pred.shape)
                # dci.dst_border_pred2gt_v[dci.border_gt] = dci.pred_border_dst[dci.border_gt]
                valid = (
                    len(dci.dst_border_pred2gt)
                    and np.inf not in dci.dst_border_pred2gt
                )

                dci.pred_hd = (
                    dci.dst_border_pred2gt_abs.max() if valid else np.nan
                )

                dci.pred_hd_avg = (
                    dci.dst_border_pred2gt_abs.mean() if valid else np.nan
                )

                dci.pred_hd95 = (
                    np.quantile(dci.dst_border_pred2gt_abs, 0.95)
                    if valid
                    else np.nan
                )

                dci.hd = np.mean([dci.gt_hd, dci.pred_hd])
                dci.hd_avg = np.mean([dci.gt_hd_avg, dci.pred_hd_avg])
                dci.hd95 = np.mean([dci.gt_hd95, dci.pred_hd95])
                # calculate HD distance}

                resc["components"][ri] = {
                    # 'detected': sum(dci.pred_comp_idx) > 0,
                    # 'uniform_gt': (1. / len(dci.pred_comp_idx)) if len(dci.pred_comp_idx) > 0 else 0,
                    # 'uniform_pred': (1. / len(dci.rel_gts)) if len(dci.rel_gts) > 0 else 0,
                    # 'maxd': dci.max_dst_gt,
                    "hd": dci.hd,
                    "hd_avg": dci.hd_avg,
                    "hd_95": dci.hd95,
                    "hd gt2pred": dci.dst_border_gt2pred_abs.mean()
                    if len(dci.dst_border_gt2pred)
                    else 0,
                    "hd pred2gt": dci.dst_border_pred2gt_abs.mean()
                    if len(dci.dst_border_pred2gt)
                    else 0,
                }
            resc["pN"] = dc.pN
            resc["gN"] = dc.gN

        if return_debug_data:
            return res, data
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
