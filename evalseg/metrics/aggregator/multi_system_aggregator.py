from . import AggregatorABS
import deepcopy


class MultiSystemAggregatorABS:

    def __init__(self, base_aggregator):
        self.base_aggregator = base_aggregator
        self.__items = {}
        pass

    def multi_add(self, pred_metric_out):
        if len(self.__items) == 0:
            self.__items = {p: deepcopy.copy(self.base_aggregator) for p in pred_metric_out}

        assert all([p in self.__items for p in pred_metric_out])
        assert all([p in pred_metric_out for p in self.__items])

        for p in self.__items:
            self.__items.add(pred_metric_out[p])

    def get(self):
        return {p: self.__items.get() for p in self.__items}
