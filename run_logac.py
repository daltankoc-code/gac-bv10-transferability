"""Leave-one-adsorbent-out validation: all observations of one GAC product are withheld in turn
(16 products). Output: robust_predictions_LOGAC.csv."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from common import load, MODELS
from paths import RESULTS; OUT = str(RESULTS)

d = load()


def run(model, prod):
    tr = d[(d['product name'] != prod) & (d.flag == 0)]
    te = d[d['product name'] == prod]
    m = MODELS[model]().fit(tr)
    out = te[['study', 'campaign', 'flag', 'y', 'scale', 'matrix_type', 'product name']].copy()
    out['pred'] = m.predict(te)
    out['model'] = model
    return out


jobs = [(m, p) for m in ['GBM', 'SAM', 'MLR', 'MEAN'] for p in sorted(d['product name'].unique())]
res = Parallel(n_jobs=2, verbose=2)(delayed(run)(*j) for j in jobs)
pd.concat(res).to_csv(f"{OUT}/robust_predictions_LOGAC.csv", index=False)
