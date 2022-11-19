from . import MetricABS, MultiMetric
import numpy as np
from ..common import parallel_runner
import copy


class MultiClassMetric(MultiMetric):
    def __init__(self, binary_metric: MetricABS, num_classes, debug={}):

        # self.binary_metric_clas = binary_metric_clas
        self.num_classes = num_classes

        super().__init__({c: binary_metric() if callable(binary_metric) else copy.deepcopy(binary_metric)
                          for c in range(1, num_classes)}, debug)
        pass

    def set_reference(self, reference: np.ndarray, spacing=None, **kwargs):
        for c in self.metrics:
            if any([self.debug[d] for d in self.debug]):
                from IPython.display import display
                display(f"======= setting reference class={c} ======")

            self.metrics[c].set_reference(reference == c, spacing=spacing, **kwargs)

    def evaluate(self, test: np.ndarray, return_debug=False, parallel=1, **kwargs):
        items = [{'metric': self.metrics[c], 'data': test == c, 'label': c, 'p':'p',
                  'return_debug':return_debug, 'kwargs':kwargs}
                 for c in self.metrics]
        parallel_res = parallel_runner(_evaluate_helper, items, parallel=parallel)
        res = {c: {} for c in self.metrics}
        debug_info = {c: {} for c in self.metrics}
        for k, v in parallel_res:
            if return_debug:
                res[k['label']], debug_info[k['label']] = v
            else:
                res[k['label']] = v
        if return_debug:
            return res, debug_info
        return res

    def evaluate_multi(self, test_dic, return_debug=False, parallel=1, **kwargs):
        items = [{'metric': self.metrics[c], 'data': test_dic[p] == c, 'p': p, 'label':c,
                  'return_debug':return_debug, 'kwargs':kwargs}
                 for p in test_dic for c in self.metrics]
        parallel_res = parallel_runner(_evaluate_helper, items, parallel=parallel)
        res = {p: {c: {} for c in self.metrics} for p in test_dic}
        debug_info = {p: {c: {} for c in self.metrics} for p in test_dic}
        for k, v in parallel_res:
            if return_debug:
                res[k['p']][k['label']], debug_info[k['p']][k['label']] = v
            else:
                res[k['p']][k['label']] = v

        if return_debug:
            return res, debug_info
        return res


def _evaluate_helper(metric, data, return_debug, label, p, **kwargs):
    if any([metric.debug[d] for d in metric.debug]):
        print(f"======= evaluate class={label} p={p} ======")
    return metric.evaluate(data, return_debug=return_debug, debug_prefix=f'c{label}{p} ', ** kwargs)
