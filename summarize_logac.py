"""Summary of leave-one-adsorbent-out validation (Section 3.3): study-weighted MAE, its ratio to the
point-level value and the fraction of twofold over-predictions, with paired cluster-bootstrap
confidence intervals over GAC products. Output: robust_logac_summary.csv."""
import glob
import numpy as np
import pandas as pd
from paths import RESULTS; OUT = str(RESULTS)

rng = np.random.default_rng(2028); B = 5000
L = pd.read_csv(f"{OUT}/robust_predictions_LOGAC.csv")
L = L[L.flag == 0].assign(e=lambda d: d.pred - d.y)
P = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")])
P = P[(P.flag == 0) & (P.scheme == 'random_point')].assign(ae=lambda d: (d.pred - d.y).abs())
pt_rep = P.groupby(['model', 'rep', 'study']).ae.mean().groupby(['model', 'rep']).mean().groupby('model').mean()
rows = []
for m in ['GBM', 'SAM', 'MLR', 'MEAN']:
    x = L[L.model == m]
    x = x.assign(ae=x.e.abs(), o2=(x.e > np.log10(2)))
    st = x.groupby('study')[['ae', 'o2']].mean()
    prods = x['product name'].unique()
    byp = {p: x[x['product name'] == p][['study', 'ae']] for p in prods}
    bs = []
    for _ in range(B):
        pick = rng.choice(prods, len(prods))
        q = pd.concat([byp[p] for p in pick])
        bs.append(q.groupby('study').ae.mean().mean())
    rows.append(dict(model=m, products=len(prods), MAE_sw=st.ae.mean(), ci_lo=np.percentile(bs, 2.5),
                     ci_hi=np.percentile(bs, 97.5), rho=st.ae.mean() / pt_rep[m], over2x_sw=st.o2.mean(),
                     bias=x.e.mean()))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/robust_logac_summary.csv", index=False)
print(R.round(3).to_string(index=False))
