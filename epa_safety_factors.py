"""Table 9 of the article (LOSO and LOCO factor columns): number of EPA breakthrough curves for which the
design volume, predicted BV10 divided by the safety factor k_tau, exceeds the upper limit of the observed
BV10 interval. k_tau is the tau-quantile of the group-weighted out-of-sample residual distribution
(equal weight per study for LOSO, equal weight per campaign for LOCO), as in eqns (10) and (11)."""
import glob
import numpy as np
import pandas as pd
from paths import RESULTS; OUT = str(RESULTS)


def wq(r, g, tau):
    w = 1.0 / pd.Series(g).map(pd.Series(g).value_counts()).values
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / w.sum()
    return r[o][np.searchsorted(cw, tau)]


P = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")])
E = pd.read_csv(f"{OUT}/epa_external.csv")
rows = []
for m in ['GBM', 'SAM', 'MLR']:
    for sch, gcol in [('LOSO', 'study'), ('LOCO', 'campaign')]:
        x = P[(P.scheme == sch) & (P.model == m) & (P.flag == 0)]
        r, g = (x.y - x.pred).values, x[gcol].values
        for tau in [0.10, 0.05]:
            q = wq(r, g, tau)
            design = E[f'pred_{m}_mid'] + q                  # log10 of the design volume
            rows.append(dict(model=m, calib=sch, tau=tau, k=round(10 ** -q, 2),
                             premature=int((design > E.yhi).sum())))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/robust_epa_loco_factor.csv", index=False)
print(R.to_string(index=False))
