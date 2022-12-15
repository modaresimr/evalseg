from . import MetricABS, MultiMetric
import numpy as np
from ..common import parallel_runner
import copy
from ..io import SegmentArray


class MultiClassMetric(MultiMetric):
    def __init__(self, binary_metric: MetricABS, num_classes, debug={}):

        # self.binary_metric_clas = binary_metric_clas
        self.num_classes = num_classes
        self.base_metric = (binary_metric if callable(binary_metric) else binary_metric.__class__).__name__
        super().__init__({c: binary_metric() if callable(binary_metric) else copy.deepcopy(binary_metric)
                          for c in range(1, num_classes)}, debug)
        pass

    def set_reference(self, reference: SegmentArray, **kwargs):
        self.reference = reference
        for c in self.metrics:
            self.metrics[c].set_reference(reference == c, **kwargs)

    def get_items_multi(self, test_dic, return_debug=False, **kwargs):
        items = super().get_items_multi(test_dic, return_debug, **kwargs)
        for item in items:
            if isinstance(item['data'], SegmentArray):
                item['data'] = item['data'] == item['metric_label']  # only for a given label
            # item['data'] = item['data'] == item['metric_label']  # only for a given label

        return items
