from . import MetricABS
import numpy as np
from .. import common, geometry, ui


class NSD(MetricABS):
    def __init__(self, debug={}, tau=1):
        super().__init__(debug)
        self.tau = tau
        pass

    def set_reference(self, reference: np.ndarray, spacing=None, **kwargs):
        super().set_reference(reference, spacing, **kwargs)
        self.gt_dst = np.abs(geometry.distance(reference, spacing=self.spacing, mode="both"))
        self.gt_border_tau = self.gt_dst < self.tau
        self.gt_border = geometry.find_binary_boundary(reference, mode="inner")

    def evaluate(self, test: np.ndarray, **kwargs):
        is2d = (test.ndim == 2 or test.shape[2] == 1)

        pr_dst = np.abs(geometry.distance(test, spacing=self.spacing, mode="both"))
        pr_border_tau = pr_dst < self.tau
        pr_border = geometry.find_binary_boundary(test, mode="inner")

        nsd = (np.sum(pr_border_tau & self.gt_border)+np.sum(pr_border & self.gt_border_tau))/(np.sum(pr_border)+np.sum(self.gt_border))

        return nsd
