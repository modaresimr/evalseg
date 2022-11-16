from ...common import dict_op
from . import AggregatorABS

epsilon = 0.000001


class Average:

    def __init__(self):
        self.__count = 0
        self.__items = {}

    def add(self, metric_out):
        if len(self.__items) == 0:
            self.__items = dict_op.multiply(metric_out, 0)
        assert dict_op.have_same_keys(metric_out, self.__items)
        self.__items = dict_op.sum(self.__items, metric_out)
        self.__count += 1

    def get(self):
        return dict_op.multiply(self.__items, 1/self.__count)


class MultiSystemAverage:

    def __init__(self):

        self.__items = {}

    def multi_add(self, pred_metric_out):
        if len(self.__items) == 0:
            self.__items = {p: Average() for p in pred_metric_out}

        assert dict_op.have_same_keys({p: 0 for p in pred_metric_out}, self.__items)

        for p in self.__items:
            self.__items.add(pred_metric_out[p])

    def get(self):
        return {p: self.__items[p].get() for p in self.__items}