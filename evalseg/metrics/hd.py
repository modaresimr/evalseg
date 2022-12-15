from . import MetricABS
import numpy as np
from .. import common, geometry, ui
from ..io import SegmentArray


class HD(MetricABS):
    def __init__(self, debug={}):
        super().__init__(debug)
        pass

    def set_reference(self, reference: SegmentArray, **kwargs):
        super().set_reference(reference, **kwargs)
        dense = reference.todense()
        self.gt_dst = np.abs(geometry.distance(dense, spacing=reference.voxelsize, mode="both"))
        self.gt_border = SegmentArray(geometry.find_binary_boundary(dense, mode="inner"))

    def evaluate(self, test: SegmentArray, **kwargs):

        densepr = test.todense()
        pr_dst = np.abs(geometry.distance(densepr, spacing=test.voxelsize, mode="both"))
        pr_border = geometry.find_binary_boundary(densepr, mode="inner")

        dst_gt2pred = self.gt_dst[pr_border]
        dst_pred2gt = pr_dst[self.gt_border.todense()]

        m_func = {'avg': np.mean,
                  'max': np.max,
                  '95th': lambda x: np.quantile(x, .95)
                  }

        dst_gt2pred = dst_gt2pred if len(dst_gt2pred) else [np.inf]
        dst_pred2gt = dst_pred2gt if len(dst_pred2gt) else [np.inf]
        return {t: np.mean([m_func[t](dst_pred2gt), m_func[t](dst_gt2pred)]) for t in m_func}
