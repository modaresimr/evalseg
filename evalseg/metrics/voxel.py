from . import MetricABS
import numpy as np


class Voxel(MetricABS):
    def __init__(self, debug={}):
        super().__init__(debug)
        pass

    def evaluate(self, test: np.ndarray, **kwargs):
        is2d = (test.ndim == 2 or test.shape[2] == 1)

        tp = self.reference & test
        fn = self.reference & (~test)
        fp = (~self.reference) & test
        tn = (~self.reference) & (~test)

        return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}
