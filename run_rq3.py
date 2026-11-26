"""RQ3: changeout volume from a lower prediction bound, evaluated by leave-one-study-out.
For each outer study: (i) GBM point plan, (ii) fixed safety factors, (iii) study-grouped conformal shift
calibrated by an inner leave-one-study-out on the training studies only, (iv) quantile GBM."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from common import load, GBM, SAM, FEATS, GBM_PARAMS, gbm_select_trees

from paths import RESULTS; OUT = str(RESULTS)
TAUS = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
d = load()
d['row'] = np.arange(len(d))
ok = d[d.flag == 0]


def wquantile(r, g, tau):
    """tau-quantile of residuals with equal total weight per study."""
    w = 1.0 / pd.Series(g).map(pd.Series(g).value_counts()).values
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / w.sum()
    return r[o][np.searchsorted(cw, tau)]


def outer(s):
    tr = ok[ok.study != s]
    te = ok[ok.study == s]
    res = te[['row', 'study', 'y', 'scale', 'matrix_type']].copy()
    # --- GBM point model and its inner LOSO residuals
    g = GBM().fit(tr)
    nT = g.n_used
    res['pred_GBM'] = g.predict(te)
    sam = SAM().fit(tr)
    res['pred_SAM'] = sam.predict(te)
    inner_r = {'GBM': [], 'SAM': []}
    inner_g = []
    for s2 in sorted(tr.study.unique()):
        a, b = tr[tr.study != s2], tr[tr.study == s2]
        gi = GBM(n_trees=nT).fit(a)
        si = SAM.__new__(SAM)
        si.alpha = sam.alpha
        from sklearn.linear_model import Ridge
        from common import _prep
        Xa = si._design(_prep(a), fit=True)
        si.m = Ridge(alpha=sam.alpha * len(a)).fit(Xa, a['y'].values)
        inner_r['GBM'].append(b['y'].values - gi.predict(b))
        inner_r['SAM'].append(b['y'].values - si.predict(b))
        inner_g.append(b['study'].values)
    ig = np.concatenate(inner_g)
    for mname in ['GBM', 'SAM']:
        r = np.concatenate(inner_r[mname])
        for t in TAUS:
            res[f'conf_{mname}_{t}'] = res[f'pred_{mname}'] + wquantile(r, ig, t)
    # --- quantile GBM (in-sample quantile, no shift calibration)
    X, y = tr[FEATS].values, tr['y'].values
    for t in [0.05, 0.1, 0.2, 0.3]:
        n = gbm_select_trees(X, y, n_splits=5, max_trees=1000, loss='quantile', alpha=t)
        q = GradientBoostingRegressor(n_estimators=n, loss='quantile', alpha=t, **GBM_PARAMS).fit(X, y)
        res[f'qgbm_{t}'] = q.predict(te[FEATS].values)
    return res


studies = sorted(ok.study.unique())
out = Parallel(n_jobs=2, verbose=5)(delayed(outer)(s) for s in studies)
pd.concat(out).to_csv(f"{OUT}/rq3_bounds.csv", index=False)
print("done")
