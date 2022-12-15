import pandas as pd


def to_df(new_dict):

    preds = list(new_dict.keys())
    metrics = list(new_dict[preds[0]].keys())
    cls = list(new_dict[preds[0]][metrics[0]].keys())
    c_res = {}
    for c in cls:

        dfs = []
        for m in metrics:
            val = new_dict[preds[0]][m][c]
            if m == 'mme':
                df = pd.DataFrame({p: {(m, k, ag): v['macro'][ag] for k, v in new_dict[p][m][c].items() for ag in ['prc', 'tpr', 'f1']}for p in preds}).T
            elif type(val) == dict:
                if 'macro' in val:
                    df = pd.DataFrame({p: {(m, k, ''): v for k, v in new_dict[p][m][c]['macro'].items()}for p in preds}).T
                else:
                    df = pd.DataFrame({p: {(m, k, ''): v for k, v in new_dict[p][m][c].items()}for p in preds}).T
            elif 'nsd' in m:
                df = pd.DataFrame({p: {('nsd', m.split(' ')[1], ''): new_dict[p][m][c]}for p in preds}).T
            else:
                df = pd.DataFrame({p: {(m, '', ''): val}for p in preds}).T

            df = df.drop(['tp', 'fp', 'fn', 'tn', 'iou', 'vs'], axis=1, errors='ignore')
            dfs.append(df)
        c_res[c] = pd.concat(dfs, axis=1)
        # final_df=final_df.loc[~final_df.index.str.startswith('2-')]

        # conv={p:{m:final_rank[p][m][c] for m in metrics} for p in preds_rank} for c in avg_rank[preds[0]][metrics[0]]}
    return c_res
