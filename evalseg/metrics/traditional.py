import pandas as pd

epsilon = 0.00001
# pylint: disable-all
# pylint: disable=all
# pylint: skip-file


def calculate_prc_tpr_f1(**kwargs):
    return calculate_prc_tpr_f1_multi({"res": kwargs})


def calculate_prc_tpr_f1_multi(dic):
    df = pd.DataFrame(dic).T
    df["prc"] = df["tp"] / (df["tp"] + df["fp"] + epsilon)
    df["tpr"] = df["tp"] / (df["tp"] + df["fn"] + epsilon)
    df["f1"] = 2 * df["tpr"] * df["prc"] / (df["tpr"] + df["prc"] + epsilon)
    return df
