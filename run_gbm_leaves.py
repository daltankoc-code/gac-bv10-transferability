"""Sensitivity of the GBM to the number of terminal nodes per tree, J in {4, 8, 16}, under the
original split and LOSO validation (number of trees selected as in the original procedure).
Output: robust_predictions_GBM_leaves.csv."""
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import common
from common import load, GBM
from paths import RESULTS; OUT = str(RESULTS)

d = load()


def run(J, scheme, key, tr_idx, te_idx):
    common.GBM_PARAMS['max_leaf_nodes'] = J
    common.GBM_PARAMS['max_depth'] = None   # tree size limited by J only
    tr = d.loc[tr_idx]; tr = tr[tr.flag == 0]
    te = d.loc[te_idx]
    m = GBM().fit(tr)
    out = te[['study', 'campaign', 'flag', 'y']].copy()
    out['pred'] = m.predict(te); out['J'] = J; out['scheme'] = scheme; out['n_trees'] = m.n_used
    return out


jobs = []
for J in [8, 16]:
    jobs.append((J, 'original', 0, d.index[d.Label == 'UV_training_set'], d.index[d.Label == 'original_test_set']))
    for s in sorted(d.study.unique()):
        jobs.append((J, 'LOSO', s, d.index[d.study != s], d.index[d.study == s]))
res = Parallel(n_jobs=2, verbose=2)(delayed(run)(*j) for j in jobs)
pd.concat(res).to_csv(f"{OUT}/robust_predictions_GBM_leaves.csv", index=False)
