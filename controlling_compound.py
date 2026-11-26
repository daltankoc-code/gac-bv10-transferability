"""Design for the controlling compound: for each adsorber run (study, water, GAC product, EBCT and
column type) with at least two compounds, the bed life is set by the compound with the smallest
BV10. Compares the compound-level safety factors of Table 7 applied to the predicted minimum with a
factor calibrated on the run-level residuals r = log10 min BV10 - log10 min predicted BV10
(equal weight per withheld group, runs weighted equally within a group).
Output: robust_controlling_compound.csv."""
import glob
import numpy as np
import pandas as pd
from paths import RESULTS; OUT = str(RESULTS)
from common import load

d = load().reset_index(drop=True)
d['row'] = np.arange(len(d))
run_key = ['study', 'matrix name', 'product name', 'EBCT', 'scenario']
d['run'] = d.groupby(run_key, sort=True).ngroup()
P = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")])
P = P[P.flag == 0].merge(d[['row', 'run']], on='row')
SF = pd.read_csv(f"{OUT}/robust_safety_factor.csv")


def wq(r, g, tau):
    w = 1.0 / pd.Series(g).map(pd.Series(g).value_counts()).values
    o = np.argsort(r); cw = np.cumsum(w[o]) / w.sum()
    return r[o][np.searchsorted(cw, tau)]


rows = []
for m in ['GBM', 'SAM']:
    for sch, gcol in [('LOSO', 'study'), ('LOCO', 'campaign')]:
        x = P[(P.scheme == sch) & (P.model == m)]
        runs = x.groupby('run').agg(n=('y', 'size'), ymin=('y', 'min'), pmin=('pred', 'min'), g=(gcol, 'first'))
        runs = runs[runs.n >= 2]
        r = (runs.ymin - runs.pmin).values
        g = runs.g.values
        w = 1.0 / pd.Series(g).map(pd.Series(g).value_counts()).values
        k_cmp = SF.query("model==@m and scheme==@sch and tau==0.1").factor.item()
        prem = (runs.pmin - np.log10(k_cmp) > runs.ymin).values
        q = wq(r, g, 0.10)
        rows.append(dict(model=m, scheme=sch, runs=len(runs), groups=len(np.unique(g)),
                         k_compound=k_cmp, premature_controlling=np.sum(w * prem) / w.sum(),
                         k_controlling=10 ** -q))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/robust_controlling_compound.csv", index=False)
print(R.round(3).to_string(index=False))
