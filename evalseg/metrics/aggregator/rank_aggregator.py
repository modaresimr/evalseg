from . import MultiSystemAggregatorABS
from ...common import dict_op
from scipy.stats import rankdata


class RankAgg:

    def __init__(self, metric_info={}):
        self.__items = {}
        self.__count = 0
        self.metric_info = metric_info
        pass

    def multi_add(self, pred_aggr):
        if len(self.__items) == 0:
            # self.preds = list(pred_aggr.keys())
            # self.classes = list(pred_aggr[self.preds[0]].keys())
            # self.metrics = list(pred_aggr[self.preds[0]][self.classes[0]].keys())
            # self.items = list(pred_aggr[self.preds[0]][self.classes[0]][self.metrics[0]].keys())
            # {p: {c: {m: {item: 0 for item in self.items} for m in self.metrics} for c in self.classes} for p in self.preds}
            self.__items = dict_op.multiply(pred_aggr, 0)

        assert dict_op.have_same_keys(self.__items, pred_aggr)

        ranked_preds = self.__get_rank(pred_aggr)
        self.__items = dict_op.sum(self.__items, ranked_preds)
        self.__count += 1

    def __get_rank(self, items):
        ranked_items = {}
        preds = list(items.keys())

        for c in items[preds[0]]:  # self.classes
            for m in items[preds[0]][c]:  # self.metrics
                coef = -1 if self.metric_info.get(m, {}).get('higher_is_better', True) else 1
                for item in items[preds[0]][c][m]:  # self.items:
                    to_rank = [items[p][c][m][item]*coef for p in preds]
                    rank = rankdata(to_rank, method='min')
                    for i, p in enumerate(preds):
                        ranked_items = rank[i]
        return ranked_items

    def get(self, mode='rank'):
        if mode == 'rank':
            return self.__get_rank(self.__items)
        if mode == 'avg':
            return dict_op.multiply(self.__items, 1/self.__count)
        raise Exception('invalid mode')

        # return {p: {c: {m: {item: self.__items[p][c][m][item]/self.__count for item in self.items} for m in self.metrics} for c in self.classes} for p in self.preds}
