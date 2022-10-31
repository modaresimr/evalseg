from collections import defaultdict

import numpy as np

from ..common import Object, parallel_runner

# pylint: disable-all
# pylint: disable=all
# pylint: skip-file


class MetricABS(Object):
    def __init__(self, debug={}):
        self.debug = defaultdict(bool, debug)
        pass

    def set_reference(self, reference: np.ndarray, spacing=None, **kwargs):
        self.reference = reference
        self.spacing = np.array(
            spacing if not (spacing is None) else [1, 1, 1]
        )
        self.helper = self.calculate_info(
            reference, self.spacing, **kwargs
        )

    def calculate_info(
        self,
        reference: np.ndarray,
        spacing: np.ndarray = None,
        **kwargs
    ):
        pass

    # def evaluate_single(
    #     self,
    #     reference: np.ndarray,
    #     test: np.ndarray,
    #     spacing: np.ndarray = None,
    #     **kwargs
    # ):
    #     self.set_reference(reference, spacing, **kwargs)
    #     return self.evaluate(test, **kwargs)

    def evaluate(self, test: np.ndarray, **kwargs):
        pass

    def evaluate_multi(self, test_dic):
        res = parallel_runner(_evaluate_helper, [{'metric': self, 'p': k, 'data': test_dic[k]} for k in test_dic])
        return {k[0]: v for k, v in res}


def _evaluate_helper(metric, data, return_debug, **kwargs):
    return metric.evaluate(data, return_debug=return_debug, **kwargs)
