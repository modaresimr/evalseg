from . import MetricABS
import numpy as np
from .. import common, geometry, ui
from ..io import SegmentArray


class NSD(MetricABS):
    def __init__(self, debug={}, tau=1):
        super().__init__(debug)
        self.tau = tau
        pass

    def set_reference(self, reference: SegmentArray, **kwargs):
        super().set_reference(reference, **kwargs)
        dense = reference.todense()
        mask_roi = geometry.calc_safe_roi(reference.shape, reference.roi, margin=20+self.tau)  # geometry.one_roi(dense, margin=20, return_index=True)
        gt_dst = np.abs(geometry.distance(dense, spacing=reference.voxelsize, mask_roi=mask_roi, mode="both"))
        self.gt_border_tau = SegmentArray(gt_dst < self.tau, mask_roi=mask_roi)
        self.gt_border = SegmentArray(geometry.find_binary_boundary(dense, mode="inner"), mask_roi=mask_roi)

    def evaluate(self, test: SegmentArray, **kwargs):
        pred_dense = test.todense()
        # mask_roi = geometry.one_roi(pred_dense, margin=20, return_index=True)
        mask_roi = geometry.calc_safe_roi(test.shape, test.roi, margin=20+self.tau)  # geometry.one_roi(dense, margin=20, return_index=True)
        pr_dst = np.abs(geometry.distance(pred_dense, spacing=test.voxelsize, mask_roi=mask_roi, mode="both"))
        pr_border_tau = SegmentArray(pr_dst < self.tau, mask_roi=mask_roi)
        pr_border = SegmentArray(geometry.find_binary_boundary(pred_dense, mode="inner"), mask_roi=mask_roi)

        a = (pr_border_tau & self.gt_border).sum()
        b = (pr_border & self.gt_border_tau).sum()
        c = pr_border.sum()+self.gt_border.sum()

        nsd = (a+b)/c if c != 0 else 0
        # nsd = (np.sum(pr_border_tau & self.gt_border)+np.sum(pr_border & self.gt_border_tau))/(np.sum(pr_border)+np.sum(self.gt_border))

        return nsd
