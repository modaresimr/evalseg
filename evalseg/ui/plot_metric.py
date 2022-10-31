import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from .. import metrics, ui


def plot_metric(ev, name=None):
    num_classes = len(ev)
    dic = {}
    for c in ev:
        df = metrics.traditional.calculate_prc_tpr_f1_multi(
            ev[c]["total"]
        )
        dic[f"class {c}"] = df[["prc", "tpr"]]
        # ui.spider_chart(df[['prc', 'tpr']], np.arange(0, 1, .2), title=name, ax=axes[i])
        display(
            pd.concat([df], keys=[name], names=[f"Class {c}"])
            .round(2)
            .drop("tn", axis=1)
        )
    ui.spider_chart_multi(dic, np.arange(0, 1, 0.2), title=name)


def plot_metric_multi(ev_dic, name=None):
    dic_of_cls = {}
    for k in ev_dic:
        ev = ev_dic[k]
        for c in ev:
            df = metrics.traditional.calculate_prc_tpr_f1_multi(ev[c]["total"])
            if c not in dic_of_cls:
                dic_of_cls[c] = {}
            dic_of_cls[c][k] = df[["prc", "tpr"]]
            # ui.spider_chart(df[['prc', 'tpr']], np.arange(0, 1, .2), title=name, ax=axes[c])
            # display(
            #     pd.concat([df], keys=[name], names=[f"Class {c}-{k}"])
            #     .round(2)
            #     .drop("tn", axis=1)
            # )
    for c in dic_of_cls:
        if c != 0:

            ui.spider_chart_multi(dic_of_cls[c], np.arange(0, 1, 0.2), title=f'{name}- class {c}')
