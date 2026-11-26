"""Robustness: GBM LOSO/LOCO with the number of trees selected by study-grouped inner CV
(instead of the random K-fold CV of the original script)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from common import load, FEATS, GBM_PARAMS, MAX_TREES

from paths import RESULTS; OUT = str(RESULTS)
d = load()
d['row'] = np.arange(len(d))


def grouped_trees(tr):
    X, y, g = tr[FEATS].values, tr['y'].values, tr['study'].values
    err = np.zeros(MAX_TREES)
    for a, b in GroupKFold(n_splits=5).split(X, y, g):
        m = GradientBoostingRegressor(n_estimators=MAX_TREES, **GBM_PARAMS).fit(X[a], y[a])
        for k, p in enumerate(m.staged_predict(X[b])):
            err[k] += np.sum((y[b] - p) ** 2)
    return int(np.argmin(err) + 1)


def run(scheme, key, tr_idx, te_idx):
    tr = d.loc[tr_idx]
    tr = tr[tr.flag == 0]
    te = d.loc[te_idx]
    n = grouped_trees(tr)
    m = GradientBoostingRegressor(n_estimators=n, **GBM_PARAMS).fit(tr[FEATS].values, tr['y'].values)
    out = te[['row', 'study', 'campaign', 'flag', 'y', 'scale', 'matrix_type', 'Label']].copy()
    out['pred'] = m.predict(te[FEATS].values)
    out['scheme'], out['rep'], out['model'], out['n_trees'] = scheme, key, 'GBM_groupedtrees', n
    return out


jobs = [('original', 0, d.index[d.Label == 'UV_training_set'], d.index[d.Label == 'original_test_set'])]
jobs += [('LOSO', int(s), d.index[d.study != s], d.index[d.study == s]) for s in sorted(d.study.unique())]
jobs += [('LOCO', int(c), d.index[d.campaign != c], d.index[d.campaign == c]) for c in sorted(d.campaign.unique())]
res = Parallel(n_jobs=2, verbose=5)(delayed(run)(*j) for j in jobs)
pd.concat(res).to_csv(f"{OUT}/robust_predictions_GBM_groupedtrees.csv", index=False)
print("done")
