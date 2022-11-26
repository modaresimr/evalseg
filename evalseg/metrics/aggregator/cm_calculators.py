import numpy as np
import copy
epsilon = 0.00001


def cm_calculate(dic, prefix):
    if type(dic) is dict:
        out = {k: cm_calculate(dic[k], prefix) for k in dic}

        if 'tp' in out:
            assert 'fn' in out and 'fp' in out
            tp, fp, fn = out['tp'], out['fp'], out['fn']
            out[prefix] = {}
            out[prefix][f'prc'] = prc = tp/(tp + fp + epsilon)
            out[prefix][f'tpr'] = tpr = tp/(tp + fn + epsilon)
            out[prefix][f"f1"] = 2 * tpr * prc / (tpr + prc + epsilon)
            out[prefix][f'iou'] = tp/(tp + fn+fp + epsilon)
            out[prefix][f'vs'] = 1 - ((np.abs(fn-fp) / (2*tp + fp + fn)) if (2*tp + fp + fn) > 0 else 0)

        return out

    return dic
