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
        df = metrics.calculate_prc_tpr_f1_multi(ev[c])
        dic[f"class {c}"] = df[["tpr", "prc"]]
        dic[f"class {c}"].columns = ['TPR', 'PRC']
        # ui.spider_chart(df[['prc', 'tpr']], np.arange(0, 1, .2), title=name, ax=axes[i])
        display(
            pd.concat([df], keys=[name], names=[f"Class {c}"])
            .round(2)
            .drop("tn", axis=1)
        )
    ui.spider_chart_multi(dic, np.arange(0, 1, 0.2), title=name, dst=dst, show=show)


def plot_metric_multi(ev_dic, name=None, dst=None, show=True, col=5, show_table=False):
    dic_of_cls = {c: {} for c in ev_dic[list(ev_dic.keys())[0]]}

    outhtml = {c: '' for c in dic_of_cls}
    for k in ev_dic:
        ev = ev_dic[k]
        for c in ev:
            df = metrics.traditional.calculate_prc_tpr_f1_multi(ev[c])

            dic_of_cls[c][k] = df[["tpr", "prc"]]
            dic_of_cls[c][k].columns = ['TPR', 'PRC']
            # ui.spider_chart(df[['prc', 'tpr']], np.arange(0, 1, .2), title=name, ax=axes[c])
            # display(
            if 'ignore it' in k:
                dic_of_cls[c][k] *= 0
            else:
                df2 = pd.concat([df], keys=[name], names=[f"Class {c}-{k}"]).round(2).drop("tn", axis=1)
                if show_table:
                    display(df2)
                outhtml[c] += df2.to_html()
            # )
    dstpng = dsthtml = None
    for c in dic_of_cls:

        if c != 0:
            if dst:
                root_ext = os.path.splitext(dst)
                dstpng = f"{root_ext[0]}-{c}{root_ext[1]}"
                dsthtml = f"{root_ext[0]}-{c}.html"
                with open(dsthtml, 'w') as f:
                    f.write(outhtml[c])
            ui.spider_chart_multi(dic_of_cls[c], np.arange(0, 1, 0.2), title=f'{name}- class {c}', dst=dstpng, show=show, col=col)


def plot_full_metric_multi(ev_dic, name=None, dst=None, show=True, col=5, show_table=False):
    dic_of_cls = {c: {} for c in ev_dic[list(ev_dic.keys())[0]]['mme']}
    descr = {c: {} for c in dic_of_cls}
    outhtml = {c: '' for c in dic_of_cls}
    for pred in ev_dic:
        ev = ev_dic[pred]

        for c in ev['mme']:
            descr[c][pred] = {m: v[c] for m, v in ev.items() if m != 'mme'}
            # df = metrics.traditional.calculate_prc_tpr_f1_multi(ev['mme'][c])
            res = pd.DataFrame({prop: pval['macro'] for prop, pval in ev['mme'][c].items()})
            # print(df)

            dic_of_cls[c][pred] = res.T[["tpr", "prc"]]  # {k: res[k] for k in ["tpr", "prc"]}
            dic_of_cls[c][pred].columns = ['TPR', 'PRC']
            # ui.spider_chart(df[['prc', 'tpr']], np.arange(0, 1, .2), title=name, ax=axes[c])
            # display(
            if 'ignore it' in pred:
                # dic_of_cls[c][pred] *= 0
                pass
            else:
                # df2 = pd.concat(pd.res, keys=[name], names=[f"Class {c}-{pred}"]).round(2).drop("tn", axis=1)
                # if show_table:
                # display(df2)
                # outhtml[c] += df2.to_html()
                pass
            # )
    dstpng = dsthtml = None
    for c in dic_of_cls:

        if c != 0:
            if dst:
                root_ext = os.path.splitext(dst)
                dstpng = f"{root_ext[0]}-{c}{root_ext[1]}"
                dsthtml = f"{root_ext[0]}-{c}.html"
                with open(dsthtml, 'w') as f:
                    f.write(outhtml[c])
            ui.spider_chart_multi(dic_of_cls[c], np.arange(0, 1, 0.2), title=f'{name}- class {c}', dst=dstpng, show=show, col=col)
    for c in descr:
        print(f'==============={c}===============')
        alldic = {}
        for appr, res in descr[c].items():
            newdict = {}
            for m, v in res.items():
                if m == 'voxel':
                    newdict[m] = {f'{x}-{d}': v[x][d] for x in ['micro', 'macro'] for d in v[x]}
                elif 'nsd' in m:
                    newdict['nsd'] = newdict.get('nsd', {})
                    newdict['nsd'][m.split(' ')[1]] = v
                else:
                    newdict[m] = v
            alldic[appr] = newdict
        # reform = {(outerKey, innerKey): values for outerKey, innerDict in newdict.items()
        #           for innerKey, values in (innerDict if type(innerDict) == dict else {'val': innerDict}).items()}
        # display(pd.DataFrame(reform))  # .round(2).rename_axis(appr))
        aa = pd.concat({k: pd.DataFrame.from_dict(v, 'index').stack() for k, v in alldic.items()}, axis=1).T.round(2)

        display(aa.sort_values(('nsd', 't=1')))

    print()
