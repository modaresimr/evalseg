from . import MetricABS
import numpy as np
from .. import common, geometry, ui


class HD(MetricABS):
    def __init__(self, debug={}):
        super().__init__(debug)
        pass

    def set_reference(self, reference: np.ndarray, spacing=None, **kwargs):
        super().set_reference(reference, spacing, **kwargs)
        self.gt_dst = np.abs(geometry.distance(reference, spacing=self.spacing, mode="both"))
        self.gt_border = geometry.find_binary_boundary(reference, mode="inner")

    def evaluate(self, test: np.ndarray, **kwargs):
        is2d = (test.ndim == 2 or test.shape[2] == 1)

        pr_dst = np.abs(geometry.distance(test, spacing=self.spacing, mode="both"))
        pr_border = geometry.find_binary_boundary(test, mode="inner")

        dst_gt2pred = self.gt_dst[pr_border]
        dst_pred2gt = pr_dst[self.gt_border]

        m_func = {'avg': np.mean,
                  'max': np.max,
                  '95th': lambda x: np.quantile(x, .95)
                  }

        return {t: np.mean([m_func[t](dst_pred2gt), m_func[t](dst_gt2pred)]) for t in m_func}
