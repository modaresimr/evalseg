import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from .. import metrics, ui


def plot_metric(ev, name=None):
    num_classes = len(ev["class"])
    dic = {}
    for i in range(num_classes):
        df = metrics.traditional.calculate_prc_tpr_f1_multi(
            ev["class"][i]["total"]
        )
        dic[f"class {i}"] = df[["prc", "tpr"]]
        # ui.spider_chart(df[['prc', 'tpr']], np.arange(0, 1, .2), title=name, ax=axes[i])
        display(
            pd.concat([df], keys=[name], names=[f"Class {i}"])
            .round(2)
            .drop("tn", axis=1)
        )
    ui.spider_chart_multi(dic, np.arange(0, 1, 0.2), title=name)
