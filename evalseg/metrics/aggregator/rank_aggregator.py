from ...common import dict_op
from scipy.stats import rankdata
from collections import defaultdict
import numpy as np


def rank(items, metric_info={}, desc=0):

    preds = list(items.keys())
    ranked_items = dict_op.multiply(items, 0)
    for m, mv in items[preds[0]].items():  # self.metrics
        for c in mv:  # self.classes
            key = (m, c)
            coef = 1 if metric_info.get(m, {}).get('higher_better', True) else -1
            coef *= 1 if desc else -1

            __rank_subdict(items, preds, key, coef, ranked_items)

    return ranked_items


def __get_item_dict(dict, key):
    ret = dict
    for k in key:
        ret = ret[k]
    return ret


def __set_item_dict(dict, key, val):
    ret = dict
    for i, k in enumerate(key):
        if i == len(key)-1:
            break
        ret = ret[k]
    ret[k] = val


def __rank_subdict(items, preds, key, coef, out):
    p = preds[0]
    val = __get_item_dict(items[p], key)
    if type(val) == dict:
        for k in val:
            __rank_subdict(items, preds, key+(k,), coef, out)
    else:
        __rank_internal(items, preds, key, coef, out)


def __rank_internal(items, preds, key, coef, out):
    to_rank = [__get_item_dict(items[p], key)*coef for p in preds]
    to_rank = [k if np.isfinite(k) else coef * -100000 for k in to_rank]
    rank = rankdata(to_rank, method='min')
    for i, p in enumerate(preds):
        __set_item_dict(out[p], key, rank[i])
