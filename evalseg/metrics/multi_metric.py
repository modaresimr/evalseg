
from . import MetricABS
import numpy as np
from .. import common, geometry, ui
from ..common import parallel_runner
from ..compress import decompress_arr
from ..io import SegmentArray


class MultiMetric(MetricABS):
    def __init__(self, metrics_dic, debug={}):
        super().__init__(debug)
        self.metrics = metrics_dic
        pass

    def set_reference(self, reference: SegmentArray, **kwargs):
        self.reference = reference
        for m in self.metrics:
            self.metrics[m].set_reference(reference, **kwargs)

    # def get_items_single(self, test, return_debug, **kwargs):
    #     # items = [{'metric': self.metrics[c], 'data': test, 'p':'p',
    #     #           'return_debug':return_debug, 'kwargs':kwargs}
    #     #          for c in self.metrics]

    #     return self.get_items_multi({'p': test}, return_debug, kwargs)

    def get_items_multi(self, test_dic, return_debug, **kwargs):
        return [{'metric_label': c, 'metric': self.metrics[c], 'p': p,
                 'return_debug':return_debug, 'kwargs':kwargs, 'data': test_dic[p]}
                for p in test_dic
                for c in self.metrics
                ]

    def evaluate(self, test: SegmentArray, return_debug=False, parallel=False, max_cpu=0, **kwargs):
        res = self.evaluate_multi({'p': test}, return_debug, parallel, max_cpu=max_cpu, **kwargs)
        if return_debug:
            return res[0]['p'], res[1]['p']
        return res['p']

    def evaluate_multi(self, test_dic, return_debug=False, parallel=False, max_cpu=0, **kwargs):
        items = self.get_items_multi(test_dic, return_debug, **kwargs)
        if len(items) == 1:
            parallel_res = [(items[0], _evaluate_helper(**items[0]))]
        else:
            parallel_res = parallel_runner(_evaluate_helper, items, parallel=parallel, max_cpu=max_cpu, silent=len(items) == 1)
        res = {p: {c: {} for c in self.metrics} for p in test_dic}
        debug_info = {p: {c: {} for c in self.metrics} for p in test_dic}
        for k, v in parallel_res:
            if return_debug:
                res[k['p']][k['metric_label']], debug_info[k['p']][k['metric_label']] = v
            else:
                res[k['p']][k['metric_label']] = v

        if return_debug:
            return res, debug_info
        return res


def _evaluate_helper(metric, data, return_debug, metric_label, p, **kwargs):
    if any([metric.debug[d] for d in metric.debug]):
        print(f"======= evaluate0 class={metric_label} p={p} ======")
    return metric.evaluate(data, return_debug=return_debug, debug_prefix=f'c{metric_label}{p} ', **kwargs)
