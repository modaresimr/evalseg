from . import MetricABS
import numpy as np
from ..io import SegmentArray


class Voxel(MetricABS):
    def __init__(self, debug={}):
        super().__init__(debug)
        pass

    def evaluate(self, test: SegmentArray, **kwargs):

        tp = (self.reference & test).sum()
        fn = (self.reference & (~test)).sum()
        fp = ((~self.reference) & test).sum()
        tn = ((~self.reference) & (~test)).sum()

        # tp = (self.reference & test).sum()
        # fn = (self.reference & (~test)).sum()
        # fp = ((~self.reference) & test).sum()
        # tn = ((~self.reference) & (~test)).sum()

        return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}
