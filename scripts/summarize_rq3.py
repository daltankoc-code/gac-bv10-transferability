import numpy as np
import pandas as pd

from paths import RESULTS; OUT = str(RESULTS)
R = pd.read_csv(f"{OUT}/rq3_bounds.csv")


def evaluate(b, y, study):
    """early: P(actual BV10 < planned changeout); unused: share of BV10 capacity left unused (linear BV)."""
    early = (y < b).astype(float)
    unused = np.clip(1 - 10 ** (b - y), 0, None)
    df = pd.DataFrame({'s': study, 'early': early, 'unused': unused})
    per = df.groupby('s').mean()
    return dict(early_studyw=per.early.mean(), early_pooled=early.mean(), unused_studyw=per.unused.mean(),
                worst_study_early=per.early.max(), n_studies_early_gt_25=int((per.early > 0.25).sum()))


rows = []
y, s = R.y.values, R.study.values
for m in ['GBM', 'SAM']:
    rows.append(dict(strategy=f'{m} point plan', level=np.nan, **evaluate(R[f'pred_{m}'].values, y, s)))
for k in [1.5, 2, 3, 5]:
    rows.append(dict(strategy='GBM / fixed safety factor', level=k, **evaluate(R['pred_GBM'].values - np.log10(k), y, s)))
for m in ['GBM', 'SAM']:
    for c in [c for c in R.columns if c.startswith(f'conf_{m}_')]:
        t = float(c.split('_')[-1])
        rows.append(dict(strategy=f'{m} + study-conformal shift', level=t, **evaluate(R[c].values, y, s)))
for c in [c for c in R.columns if c.startswith('qgbm_')]:
    rows.append(dict(strategy='quantile GBM', level=float(c.split('_')[-1]), **evaluate(R[c].values, y, s)))
T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/table_rq3_tradeoff.csv", index=False)
print(T.round(3).to_string(index=False))

# implied safety factor of the conformal shift at tau=0.1 per outer fold
for m in ['GBM', 'SAM']:
    sh = (R[f'pred_{m}'] - R[f'conf_{m}_0.1']).groupby(R.study).first()
    print(m, "tau=0.1 shift (log10):", sh.round(3).describe()[['min', '50%', 'max']].to_dict(),
          " -> factor", (10 ** sh).describe()[['min', '50%', 'max']].round(2).to_dict())
