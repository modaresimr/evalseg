import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
import os
from .. import metrics, ui


def plot_metric(ev, name=None, dst=None, show=True):
    num_classes = len(ev)
    dic = {}
    for c in ev:
        df = metrics.calculate_prc_tpr_f1_multi(ev[c]["total"])
        dic[f"class {c}"] = df[["prc", "tpr"]]
        # ui.spider_chart(df[['prc', 'tpr']], np.arange(0, 1, .2), title=name, ax=axes[i])
        display(
            pd.concat([df], keys=[name], names=[f"Class {c}"])
            .round(2)
            .drop("tn", axis=1)
        )
    ui.spider_chart_multi(dic, np.arange(0, 1, 0.2), title=name, dst=dst, show=show)


def plot_metric_multi(ev_dic, name=None, dst=None, show=True,col=5):
    dic_of_cls = {c: {} for c in ev_dic[list(ev_dic.keys())[0]]}

    outhtml = {c: '' for c in dic_of_cls}
    for k in ev_dic:
        ev = ev_dic[k]
        for c in ev:
            df = metrics.traditional.calculate_prc_tpr_f1_multi(ev[c]["total"])

            dic_of_cls[c][k] = df[["prc", "tpr"]]
            # ui.spider_chart(df[['prc', 'tpr']], np.arange(0, 1, .2), title=name, ax=axes[c])
            # display(
            if 'ignore it' in k:
                dic_of_cls[c][k] *= 0
            else:
                outhtml[c] += pd.concat([df], keys=[name], names=[f"Class {c}-{k}"]).round(2).drop("tn", axis=1).to_html()
            # )
    root_ext = os.path.splitext(dst)
    for c in dic_of_cls:
        dstpng = f"{root_ext[0]}-{c}{root_ext[1]}"
        dsthtml = f"{root_ext[0]}-{c}.html"
        if c != 0:
            with open(dsthtml, 'w') as f:
                f.write(outhtml[c])
            ui.spider_chart_multi(dic_of_cls[c], np.arange(0, 1, 0.2), title=f'{name}- class {c}', dst=dstpng, show=show,col=col)
