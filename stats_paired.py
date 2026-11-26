"""Paired cluster bootstrap for the error-inflation ratio rho (numerator and denominator from the
same resampled studies), paired differences of rho between models, TOST equivalence test for the
SAM-GBM LOSO comparison, and robust error summaries (median and 90th percentile of |e|, alternative
study weightings)."""
import glob
import numpy as np
import pandas as pd
from paths import RESULTS; OUT = str(RESULTS)

rng = np.random.default_rng(2027); B = 5000
P = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")])
P = P[P.flag == 0].assign(e=lambda d: d.pred - d.y)
P['ae'] = P.e.abs()
MODELS = ['GBM', 'SAM', 'MLR', 'MEAN', 'SAM_NOINT']

# per-study point-level MAE averaged over the replicates in which the study is in the test set
pt = P[P.scheme == 'random_point'].groupby(['model', 'rep', 'study']).ae.mean().groupby(['model', 'study']).mean()
pt_rep = P[P.scheme == 'random_point'].groupby(['model', 'rep', 'study']).ae.mean().groupby(['model', 'rep']).mean().groupby('model').mean()

camp_of = P.drop_duplicates('study').set_index('study').campaign
studies = np.sort(P.study.unique())
units = {'LOSO': studies, 'LOCO': np.sort(P.campaign.unique())}
picks = {k: [rng.choice(v, len(v)) for _ in range(B)] for k, v in units.items()}


def studies_in(scheme, pick):
    if scheme == 'LOSO':
        return pick
    return np.concatenate([camp_of.index[camp_of.values == c].values for c in pick])


rows, rho_bs = [], {}
for m in MODELS:
    for sch in ['LOSO', 'LOCO']:
        num = P[(P.scheme == sch) & (P.model == m)].groupby('study').ae.mean()
        den = pt.loc[m]
        rho = num.mean() / pt_rep[m]
        bs = []
        for pick in picks[sch]:
            s = studies_in(sch, pick)
            bs.append(num.loc[s].mean() / den.loc[s].mean())
        bs = np.array(bs); rho_bs[(m, sch)] = bs
        rows.append(dict(model=m, scheme=sch, MAE_sw=num.mean(), MAE_sw_point=pt_rep[m], rho=rho,
                         rho_lo=np.percentile(bs, 2.5), rho_hi=np.percentile(bs, 97.5)))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/robust_ratio_paired.csv", index=False)
print(R.round(3).to_string(index=False))

rows = []
for a, b in [('GBM', 'SAM'), ('GBM', 'MLR'), ('SAM', 'MLR')]:
    for sch in ['LOSO', 'LOCO']:
        d = rho_bs[(a, sch)] - rho_bs[(b, sch)]
        est = R.query("model==@a and scheme==@sch").rho.item() - R.query("model==@b and scheme==@sch").rho.item()
        rows.append(dict(comparison=f"{a}-{b}", scheme=sch, diff=est, ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5)))
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/robust_rho_diff.csv", index=False)
print(D.round(3).to_string(index=False))

# TOST: SAM - GBM study-level LOSO MAE, margin +-0.1 log units, 90% percentile bootstrap interval
L = P[P.scheme == 'LOSO'].groupby(['study', 'model']).ae.mean().unstack()
rows = []
for a, b in [('SAM', 'GBM'), ('MLR', 'GBM'), ('MLR', 'SAM')]:
    diff = (L[a] - L[b]).values
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [5, 95])
    rows.append(dict(comparison=f"{a}-{b}", mean_diff=diff.mean(), ci90_lo=lo, ci90_hi=hi, margin=0.1,
                     equivalent=bool(lo > -0.1 and hi < 0.1)))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/robust_tost.csv", index=False)
print(T.round(3).to_string(index=False))

# robust summaries and weighting sensitivity
rows = []
for m in MODELS:
    for sch in ['original', 'random_point', 'LOSO', 'LOCO']:
        x = P[(P.scheme == sch) & (P.model == m)]
        if sch == 'random_point':
            med = x.groupby('rep').ae.median().mean(); p90 = x.groupby('rep').ae.quantile(0.9).mean()
            sw = sq = big = np.nan
        else:
            med, p90 = x.ae.median(), x.ae.quantile(0.9)
            g = x.groupby('study').ae.agg(['mean', 'size'])
            sq = np.average(g['mean'], weights=np.sqrt(g['size']))
            big = g.loc[g['size'] >= 5, 'mean'].mean()
        rows.append(dict(model=m, scheme=sch, median_ae=med, p90_ae=p90, MAE_sqrtn=sq, MAE_sw_n5=big))
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/robust_summaries.csv", index=False)
print(S.round(3).to_string(index=False))
