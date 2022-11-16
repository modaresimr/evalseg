
from . import AggregatorABS
from ...common import dict_op
epsilon = 0.000001


class Macro(AggregatorABS):

    def __init__(self):
        self.__count = 0
        self.__items = {}

    def add(self, metric_out):
        if len(self.__items) == 0:
            self.__items = {m: {item: 0 for item in ['tpr', 'prc', 'f1']} for m in metric_out}

        for m in self.__items:
            new_items = {}
            assert all([item in metric_out[m] for item in ['tp', 'fp', 'fn']])

            new_items['tpr'] = metric_out[m]['tp']/(metric_out[m]['tp']+metric_out[m]['fn']+epsilon)
            new_items['prc'] = metric_out[m]['tp']/(metric_out[m]['tp']+metric_out[m]['fp']+epsilon)
            new_items['f1'] = 2*new_items['prc']*new_items['tpr']/(new_items['prc']+new_items['tpr']+epsilon)
            assert dict_op.have_same_keys(self.__items, new_items)
            assert all([new_items[item] >= 0 for item in new_items])
            self.__items[m] = dict_op.sum(self.__items[m], new_items)

        self.__count += 1

    def get(self):
        # return {m: self.__items[m]/self.__count for m in self.__items}
        return dict_op.multiply(self.__items, 1/self.__count)
