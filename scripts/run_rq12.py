"""RQ1/RQ2: original split vs point-level random splits vs leave-one-study-out vs leave-one-campaign-out,
for GBM (as published), MLR and the structured additive model (SAM). Saves out-of-sample predictions."""
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ShuffleSplit
from common import load, MODELS

from paths import RESULTS; OUT = str(RESULTS)
d = load()
d['row'] = np.arange(len(d))
train_ok = d['flag'] == 0          # flagged rows are never used for training (as in the original study)


def run(scheme, rep, model, tr_idx, te_idx):
    tr = d.loc[tr_idx]
    tr = tr[tr['flag'] == 0]
    te = d.loc[te_idx]
    m = MODELS[model]().fit(tr)
    out = te[['row', 'study', 'campaign', 'flag', 'y', 'scale', 'matrix_type', 'Label']].copy()
    out['pred'] = m.predict(te)
    out['scheme'], out['rep'], out['model'] = scheme, rep, model
    out['n_trees'] = getattr(m, 'n_used', np.nan)
    return out


jobs = []
models = sys.argv[1].split(',') if len(sys.argv) > 1 else ['GBM', 'MLR', 'SAM']
for model in models:
    # S1: published split
    jobs.append(('original', 0, model, d.index[d.Label == 'UV_training_set'], d.index[d.Label == 'original_test_set']))
    # S2: repeated random point-level 9:1 splits
    for r, (tr, te) in enumerate(ShuffleSplit(n_splits=20, test_size=0.1, random_state=7).split(d)):
        jobs.append(('random_point', r, model, d.index[tr], d.index[te]))
    # S3: leave-one-study-out
    for s in sorted(d.study.unique()):
        jobs.append(('LOSO', int(s), model, d.index[d.study != s], d.index[d.study == s]))
    # S4: leave-one-campaign-out
    for c in sorted(d.campaign.unique()):
        jobs.append(('LOCO', int(c), model, d.index[d.campaign != c], d.index[d.campaign == c]))

res = Parallel(n_jobs=2, verbose=5)(delayed(run)(*j) for j in jobs)
pd.concat(res).to_csv(f"{OUT}/predictions_{'_'.join(models)}.csv", index=False)
print("done", len(jobs))
