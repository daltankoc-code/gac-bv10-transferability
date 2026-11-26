"""Sensitivity of the EPA external test to (i) retaining the first (day-0) sample of each curve,
(ii) DOC taken as 0.9 or 0.8 times the reported TOC, and (iii) the EBCT set to the minimum of the
database (4.6 min), and (iv) 13 days of operation of the GAC columns before the first sample, the
offset of the bed-volume axis of the ion-exchange columns in sheet Fig2 of the same workbook. Reports, for each model, the number of curves whose prediction exceeds the upper
limit of the observed interval, the median lower bound of the over-prediction factor, and the number
of curves for which the design volume with the LOCO factor at tau = 0.10 exceeds that limit.
Output: robust_epa_sensitivity.csv."""
import numpy as np
import pandas as pd
from common import load, GBM, MLR, SAM
from epa_external import read_curves, TRAIN, F400
from paths import RESULTS; OUT = str(RESULTS)

K = pd.read_csv(f"{OUT}/robust_epa_loco_factor.csv").query("calib=='LOCO' and tau==0.1").set_index('model').k
K5 = pd.read_csv(f"{OUT}/robust_epa_loco_factor.csv").query("calib=='LOCO' and tau==0.05").set_index('model').k


def interval(days, inf, eff, ebct, first, prior=0.0):
    bv = (days + prior) * 1440.0 / ebct
    ratio = eff / np.nanmean(inf)
    start = 0 if first else 1
    for k in range(start, len(days)):
        nxt = ratio[k + 1] if k + 1 < len(days) else ratio[k]
        if ratio[k] >= 0.1 and nxt >= 0.1:
            lo = bv[k - 1] if k - 1 >= start else prior * 1440.0 / ebct
            return lo, bv[k]
    return bv[-1], np.inf


d = load(); ok = d[d.flag == 0]
abr = d.drop_duplicates('Name of compound').set_index('Name of compound')[['E', 'S', 'A', 'B', 'V', 'L', 'charge']]
gw = d[d.matrix_type.str.contains('ground', case=False, na=False) & (d.DOC > 0)]
suva = (gw.UV254 * 100 / gw.DOC); suva_med = suva[suva > 0].median()
models = {n: M().fit(ok) for n, M in [('GBM', GBM), ('SAM', SAM), ('MLR', MLR)]}
variants = {'baseline': dict(first=False, doc=1.0, ebct=None),
            'prior operation of 13 days': dict(first=False, doc=1.0, ebct=None, prior=13.0),
            'first sample retained': dict(first=True, doc=1.0, ebct=None),
            'DOC = 0.9 TOC': dict(first=False, doc=0.9, ebct=None),
            'DOC = 0.8 TOC': dict(first=False, doc=0.8, ebct=None),
            'EBCT = 4.6 min': dict(first=False, doc=1.0, ebct=4.6)}
rows = []
for vname, v in variants.items():
    recs = []
    for tname, sheet in [('preAS', 'Fig3_preAS'), ('postAS', 'Fig3_postAS')]:
        comps, days, inf, eff = read_curves(sheet)
        for j, c in enumerate(comps):
            lo, hi = interval(days, inf[:, j], eff[:, j], TRAIN[tname]['EBCT'], v['first'], v.get('prior', 0.0))
            r = dict(compound=c, C0=np.nanmean(inf[:, j]) / 1000.0, BV10_hi=hi, **TRAIN[tname], **F400)
            r.update(abr.loc[c].to_dict())
            r['DOC'] = TRAIN[tname]['DOC'] * v['doc']
            if v['ebct']:
                r['EBCT'] = v['ebct']
            recs.append(r)
    E = pd.DataFrame(recs)
    E['UV254'] = suva_med * E.DOC / 100
    yhi = np.log10(E.BV10_hi.clip(lower=1e-9))
    for n, m in models.items():
        p = m.predict(E)
        rows.append(dict(variant=vname, model=n, above_upper=int((p > yhi).sum()),
                         median_lb_factor=float(10 ** np.median(p - yhi)) if np.all(np.isfinite(yhi)) else float(10 ** np.median((p - yhi)[E.BV10_hi > 0])),
                         premature_LOCO_k010=int((p - np.log10(K[n]) > yhi).sum()),
                         premature_LOCO_k005=int((p - np.log10(K5[n]) > yhi).sum())))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/robust_epa_sensitivity.csv", index=False)
print(R.round(2).to_string(index=False))
