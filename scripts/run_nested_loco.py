"""Campaign-level nested calibration of the safety factor.
Outer loop: leave one campaign out. Inner loops on the remaining campaigns:
(i) leave-one-campaign-out residuals -> q_tau (campaign-weighted),
(ii) leave-one-study-out residuals   -> q_tau (study-weighted).
Both shifts are applied to the outer prediction of the withheld campaign."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from common import load, GBM, SAM, _prep

from paths import RESULTS; OUT = str(RESULTS)
TAUS = [0.05, 0.10, 0.20]
d = load()
d['row'] = np.arange(len(d))
ok = d[d.flag == 0]


def wquantile(r, g, tau):
    w = 1.0 / pd.Series(g).map(pd.Series(g).value_counts()).values
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / w.sum()
    return r[o][np.searchsorted(cw, tau)]


def sam_fixed(train, alpha):
    s = SAM.__new__(SAM)
    s.alpha = alpha
    X = s._design(_prep(train), fit=True)
    s.m = Ridge(alpha=alpha * len(train)).fit(X, train['y'].values)
    return s


def outer(c):
    tr = ok[ok.campaign != c]
    te = ok[ok.campaign == c]
    res = te[['row', 'study', 'campaign', 'y', 'scale']].copy()
    g = GBM().fit(tr)
    s = SAM().fit(tr)
    res['pred_GBM'], res['pred_SAM'] = g.predict(te), s.predict(te)
    for unit in ['campaign', 'study']:
        r = {'GBM': [], 'SAM': []}
        grp = []
        for u in sorted(tr[unit].unique()):
            a, b = tr[tr[unit] != u], tr[tr[unit] == u]
            r['GBM'].append(b['y'].values - GBM(n_trees=g.n_used).fit(a).predict(b))
            r['SAM'].append(b['y'].values - sam_fixed(a, s.alpha).predict(b))
            grp.append(b[unit].values)
        grp = np.concatenate(grp)
        for m in ['GBM', 'SAM']:
            rr = np.concatenate(r[m])
            for t in TAUS:
                res[f'b_{m}_{unit}_{t}'] = res[f'pred_{m}'] + wquantile(rr, grp, t)
    return res


camps = sorted(ok.campaign.unique())
out = Parallel(n_jobs=2, verbose=5)(delayed(outer)(c) for c in camps)
pd.concat(out).to_csv(f"{OUT}/nested_loco_bounds.csv", index=False)
print("done")
