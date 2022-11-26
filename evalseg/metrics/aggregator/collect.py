from ...common import dict_op
import numpy as np
epsilon = 0.000001


class Collect:

    def __init__(self):
        self.__items = {}

    def add(self, metric_out):
        if len(self.__items) == 0:
            self.__items = dict_op.apply_func(metric_out, lambda s: ())
        # dict_op.assert_same_keys(self.__items, metric_out)
        # self.__items, metric_out = dict_op.common_keys(self.__items, metric_out)

        self.__items = dict_op.concat(self.__items, metric_out)

    def get(self, func=np.array):
        return dict_op.apply_func(self.__items, func)
