"""Robustness checks: (1) study-cluster bootstrap CIs, (3) safety factor implied by LOSO residuals,
(4) applicability domain of the EPA columns, (5) sensitivity to flagged observations."""
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from common import load, metrics, FEATS

from paths import RESULTS; OUT = str(RESULTS)
rng = np.random.default_rng(2026)
B = 5000
P = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")])
P['err'] = P.pred - P.y


def per_group(df, g):
    return df.groupby(g).agg(ae=('err', lambda e: e.abs().mean()), n=('err', 'size'),
                             over2=('err', lambda e: (e > np.log10(2)).sum()))


def boot_mean(v):
    v = np.asarray(v)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return np.percentile(bs, [2.5, 97.5])


# ---------- (1) bootstrap CIs
rows = []
for m in ['GBM', 'SAM', 'MLR']:
    rnd = P[(P.scheme == 'random_point') & (P.model == m) & (P.flag == 0)]
    rnd_rep = rnd.groupby('rep').apply(lambda x: per_group(x, 'study')['ae'].mean())
    for sch, g in [('LOSO', 'study'), ('LOCO', 'campaign')]:
        x = P[(P.scheme == sch) & (P.model == m) & (P.flag == 0)]
        pg = per_group(x, g)
        est = pg['ae'].mean()
        lo, hi = boot_mean(pg['ae'])
        # ratio to random-split error: bootstrap groups for numerator, reps for denominator
        num = pg['ae'].values[rng.integers(0, len(pg), (B, len(pg)))].mean(1)
        den = rnd_rep.values[rng.integers(0, len(rnd_rep), B)]
        rows.append(dict(model=m, scheme=sch, groups=len(pg), MAE_groupw=est, ci_lo=lo, ci_hi=hi,
                         random_MAE=rnd_rep.mean(), ratio=est / rnd_rep.mean(),
                         ratio_ci_lo=np.percentile(num / den, 2.5), ratio_ci_hi=np.percentile(num / den, 97.5)))
CI = pd.DataFrame(rows)
CI.to_csv(f"{OUT}/robust_bootstrap_ci.csv", index=False)
print("(1) cluster-bootstrap CIs\n", CI.round(3).to_string(index=False))

# pilot/full over-prediction > 2x (cluster bootstrap over studies containing pilot/full points)
lines = []
for m in ['GBM', 'SAM', 'MLR']:
    for sch in ['random_point', 'LOSO', 'LOCO']:
        x = P[(P.scheme == sch) & (P.model == m) & (P.flag == 0) & (P.scale == 'pilot/full')]
        if sch == 'random_point':
            r = x.groupby('rep').apply(lambda z: (z.err > np.log10(2)).mean())
            lines.append(dict(model=m, scheme=sch, rate=r.mean(), ci_lo=np.percentile(r, 2.5), ci_hi=np.percentile(r, 97.5), n_points=len(x) / 20))
            continue
        pg = x.groupby('study').agg(k=('err', lambda e: (e > np.log10(2)).sum()), n=('err', 'size'))
        idx = rng.integers(0, len(pg), (B, len(pg)))
        bs = pg['k'].values[idx].sum(1) / pg['n'].values[idx].sum(1)
        lines.append(dict(model=m, scheme=sch, rate=pg.k.sum() / pg.n.sum(), ci_lo=np.percentile(bs, 2.5),
                          ci_hi=np.percentile(bs, 97.5), n_points=pg.n.sum()))
PF = pd.DataFrame(lines)
PF.to_csv(f"{OUT}/robust_pilot_overprediction_ci.csv", index=False)
print("\npilot/full over-prediction >2x\n", PF.round(3).to_string(index=False))


# ---------- (3) safety factor implied by LOSO residuals
def wq(r, g, tau):
    w = 1.0 / pd.Series(g).map(pd.Series(g).value_counts()).values
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / w.sum()
    return r[o][np.searchsorted(cw, tau)]


sf = []
for m in ['GBM', 'SAM']:
    for sch in ['LOSO', 'LOCO']:
        x = P[(P.scheme == sch) & (P.model == m) & (P.flag == 0)]
        gcol = 'study' if sch == 'LOSO' else 'campaign'
        r, g = (x.y - x.pred).values, x[gcol].values
        groups = np.unique(g)
        for tau in [0.05, 0.10, 0.20]:
            q = wq(r, g, tau)
            bs = []
            for _ in range(2000):
                pick = rng.choice(groups, len(groups))
                rr = np.concatenate([r[g == k] for k in pick])
                gg = np.concatenate([np.full((g == k).sum(), j) for j, k in enumerate(pick)])
                bs.append(wq(rr, gg, tau))
            bs = np.array(bs)
            sf.append(dict(model=m, scheme=sch, tau=tau, quantile_log=q, factor=10 ** -q,
                           factor_ci_lo=10 ** -np.percentile(bs, 97.5), factor_ci_hi=10 ** -np.percentile(bs, 2.5)))
SF = pd.DataFrame(sf)
SF.to_csv(f"{OUT}/robust_safety_factor.csv", index=False)
print("\n(3) safety factor (divide predicted BV10 by factor)\n", SF.round(3).to_string(index=False))

# ---------- (5) flagged observations included
fl = []
for m in ['GBM', 'SAM', 'MLR']:
    for sch in ['original', 'LOSO', 'LOCO']:
        x = P[(P.scheme == sch) & (P.model == m)]
        a, b = metrics(x[x.flag == 0]), metrics(x)
        fl.append(dict(model=m, scheme=sch, n_unflagged=a['n'], MAE_studyw_unflagged=a['MAE_studyw'], over2x_unflagged=a['over2x_rate'],
                       n_all=b['n'], MAE_studyw_all=b['MAE_studyw'], over2x_all=b['over2x_rate']))
FL = pd.DataFrame(fl)
FL.to_csv(f"{OUT}/robust_flag_sensitivity.csv", index=False)
print("\n(5) flag sensitivity\n", FL.round(3).to_string(index=False))

# ---------- (4) applicability domain of EPA columns
d = load()
ok = d[d.flag == 0]
E = pd.read_csv(f"{OUT}/epa_external.csv")
dom = []
for f in FEATS:
    lo, hi = ok[f].min(), ok[f].max()
    emin, emax = E[f].min(), E[f].max()
    pct = (ok[f] <= emin).mean() * 100
    dom.append(dict(feature=f, train_min=lo, train_max=hi, EPA_min=emin, EPA_max=emax,
                    outside=bool(emin < lo or emax > hi), EPA_min_train_percentile=pct))
DOM = pd.DataFrame(dom)
DOM.to_csv(f"{OUT}/robust_epa_domain_ranges.csv", index=False)
print("\n(4) EPA feature ranges\n", DOM.round(3).to_string(index=False))

# multivariate: nearest-neighbour distance in standardised log-feature space
def Z(df):
    z = df[FEATS].astype(float).copy()
    for v in ['C0', 'DOC', 'UV254', 'EBCT']:
        z[v] = np.log10(z[v].clip(lower=1e-4))
    return z.values
sc = StandardScaler().fit(Z(ok))
Xt = sc.transform(Z(ok))
st = ok['study'].values
# reference: each training row's NN distance to rows of *other* studies (what LOSO faces)
ref = np.array([np.min(np.linalg.norm(Xt[st != st[i]] - Xt[i], axis=1)) for i in range(len(Xt))])
Xe = sc.transform(Z(E))
de = np.array([np.min(np.linalg.norm(Xt - x, axis=1)) for x in Xe])
E['nn_dist'] = de
E['nn_dist_pct_vs_LOSO'] = [(ref <= v).mean() * 100 for v in de]
print("\nLOSO reference NN distance: median %.2f, 95th pct %.2f" % (np.median(ref), np.percentile(ref, 95)))
print(E[['train', 'compound', 'nn_dist', 'nn_dist_pct_vs_LOSO']].round(2).to_string(index=False))
# same with EBCT set to the training minimum (isolates the EBCT contribution)
E2 = E.copy(); E2['EBCT'] = ok.EBCT.min()
de2 = np.array([np.min(np.linalg.norm(Xt - x, axis=1)) for x in sc.transform(Z(E2))])
print("median NN distance EPA: %.2f ; with EBCT clipped to training minimum: %.2f" % (np.median(de), np.median(de2)))
E[['train', 'compound', 'nn_dist', 'nn_dist_pct_vs_LOSO']].to_csv(f"{OUT}/robust_epa_domain_nn.csv", index=False)
pd.DataFrame({'ref_median': [np.median(ref)], 'ref_p95': [np.percentile(ref, 95)], 'EPA_median': [np.median(de)],
              'EPA_median_EBCTclipped': [np.median(de2)]}).to_csv(f"{OUT}/robust_epa_domain_summary.csv", index=False)
