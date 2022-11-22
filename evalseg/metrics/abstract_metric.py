from collections import defaultdict

import numpy as np
from ..io import SegmentArray
from ..common import Object, parallel_runner

# pylint: disable-all
# pylint: disable=all
# pylint: skip-file


class MetricABS(Object):
    def __init__(self, debug={}, name=None):
        super().__init__(name)
        self.debug = defaultdict(bool, debug)
        pass

    def set_reference_numpy(self, reference: np.ndarray, spacing=None, **kwargs):
        self.set_reference(SegmentArray(reference, spacing))

    def set_reference(self, reference: SegmentArray, **kwargs):
        self.reference = reference

    # def evaluate_single(
    #     self,
    #     reference: np.ndarray,
    #     test: np.ndarray,
    #     spacing: np.ndarray = None,
    #     **kwargs
    # ):
    #     self.set_reference(reference, spacing, **kwargs)
    #     return self.evaluate(test, **kwargs)

    def evaluate(self, test: SegmentArray, **kwargs):
        pass

    def evaluate_numpy(self, test: np.ndarray, **kwargs):
        return self.evaluate(SegmentArray(test, self.reference.voxelsize))

    def evaluate_multi(self, test_dic, parallel=False, max_cpu=0, **kwargs):
        res = parallel_runner(_evaluate_helper,
                              [{'metric': self, 'p': k, 'data': test_dic[k], 'kwargs':kwargs}for k in test_dic],
                              parallel=parallel, max_cpu=max_cpu)
        return {k[0]: v for k, v in res}

    def evaluate_multi_numpy(self, test_dic, **kwargs):
        return self.evaluate_multi({k: SegmentArray(test_dic[k], self.reference.voxelsize) for k in test_dic}, **kwargs)


def _evaluate_helper(metric, data, return_debug, kwargs):
    return metric.evaluate(data, return_debug=return_debug, **kwargs)
