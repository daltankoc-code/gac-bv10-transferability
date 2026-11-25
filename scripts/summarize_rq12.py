import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from common import metrics

from paths import RESULTS; OUT = str(RESULTS)
P = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")])
P = P[P.flag == 0]          # evaluate on unflagged observations

rows = []
for (sch, mod), g in P.groupby(['scheme', 'model']):
    if sch == 'random_point':
        ms = pd.DataFrame([metrics(gg) for _, gg in g.groupby('rep')])
        m = ms.mean().to_dict()
        m['MAE_sd'] = ms['MAE'].std()
    else:
        m = metrics(g)
    rows.append(dict(scheme=sch, model=mod, **m))
T = pd.DataFrame(rows)
order = {'original': 0, 'random_point': 1, 'LOSO': 2, 'LOCO': 3}
T = T.sort_values(['model', 'scheme'], key=lambda s: s.map(order) if s.name == 'scheme' else s)
T.to_csv(f"{OUT}/table_rq12_metrics.csv", index=False)
print(T.round(3).to_string(index=False))

# per-study LOSO errors
L = P[P.scheme == 'LOSO'].assign(err=lambda x: x.pred - x.y)
per = L.groupby(['study', 'model']).agg(n=('y', 'size'), MAE=('err', lambda e: e.abs().mean()),
                                        bias=('err', 'mean'), over2x=('err', lambda e: (e > np.log10(2)).mean())).unstack('model')
meta = L.drop_duplicates('study').set_index('study')[['scale', 'matrix_type']]
scale_mix = L[L.model == 'GBM'].groupby('study')['scale'].agg(lambda s: '/'.join(sorted(set(s))))
per[('meta', 'scale')] = scale_mix
per[('meta', 'matrix')] = meta['matrix_type']
per.to_csv(f"{OUT}/table_loso_per_study.csv")
print(per.round(3).to_string())

# grouping by matrix type and scale (study-weighted)
for key in ['matrix_type', 'scale']:
    gg = L.groupby([key, 'model', 'study'])['err'].apply(lambda e: e.abs().mean()).groupby([key, 'model']).agg(['mean', 'count']).unstack('model')
    print("\nLOSO study-weighted MAE by", key); print(gg.round(3).to_string())

# paired comparison across studies (RQ2)
mae = L.groupby(['study', 'model'])['err'].apply(lambda e: e.abs().mean()).unstack()
rng = np.random.default_rng(1)
lines = []
for a, b in [('SAM', 'GBM'), ('SAM', 'MLR'), ('GBM', 'MLR')]:
    diff = (mae[a] - mae[b]).values
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    p = wilcoxon(mae[a], mae[b]).pvalue
    lines.append(dict(comparison=f"{a}-{b}", mean_diff=diff.mean(), ci_lo=np.percentile(boots, 2.5),
                      ci_hi=np.percentile(boots, 97.5), wins=int((diff < 0).sum()), n=len(diff), wilcoxon_p=p))
C = pd.DataFrame(lines)
C.to_csv(f"{OUT}/table_rq2_paired.csv", index=False)
print(C.round(4).to_string(index=False))
# same for LOCO
Lc = P[P.scheme == 'LOCO'].assign(err=lambda x: x.pred - x.y)
maec = Lc.groupby(['campaign', 'model'])['err'].apply(lambda e: e.abs().mean()).unstack()
print("\nLOCO campaign-weighted MAE:", maec.mean().round(3).to_dict())
