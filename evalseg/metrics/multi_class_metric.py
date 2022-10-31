from . import MetricABS
import numpy as np
from ..common import parallel_runner


class MultiClassMetric(MetricABS):
    def __init__(self, binary_metric_clas: MetricABS, num_classes, debug={}):
        super().__init__(debug)
        self.binary_metric_clas = binary_metric_clas
        self.num_classes = num_classes
        self.metrics = {c: binary_metric_clas(debug=debug) for c in range(num_classes)}

        pass

    def set_reference(self, reference: np.ndarray, spacing=None, **kwargs):
        for c in self.metrics:
            if any([self.debug[d] for d in self.debug]):
                print(f"======= setting reference class={c} ======")

            m = self.metrics[c]
            m.set_reference(reference == c, spacing=None, **kwargs)

    def evaluate(self, test: np.ndarray, return_debug=False, **kwargs):
        items = [{'metric': self.metrics[c], 'data': test, 'class': c,
                  'return_debug':return_debug, 'kwargs':kwargs}
                 for c in self.metrics]
        parallel_res = parallel_runner(_evaluate_helper, items)
        res = {c: {} for c in self.metrics}

        for k, v in parallel_res:
            res[k['class']] = v

        return res

    def evaluate_multi(self, test_dic, return_debug=False, **kwargs):
        items = [{'metric': self.metrics[c], 'data': test_dic[p], 'p': p, 'class':c,
                  'return_debug':return_debug, 'kwargs':kwargs}
                 for p in test_dic for c in self.metrics]
        parallel_res = parallel_runner(_evaluate_helper, items)
        res = {p: {c: {} for c in self.metrics} for p in test_dic}
        debug_info = {p: {c: {} for c in self.metrics} for p in test_dic}
        for k, v in parallel_res:
            if return_debug:
                res[k['p']][k['c']], debug_info[k['p']][k['c']] = v
            else:
                res[k['p']][k['c']] = v

        if return_debug:
            return res, debug_info
        return res


def _evaluate_helper(metric, data, return_debug, **kwargs):
    return metric.evaluate(data, return_debug=return_debug, **kwargs)
