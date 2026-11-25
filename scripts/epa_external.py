"""External test on the EPA Haupert et al. (2023) F400 pilot columns (pre- and post-air-stripper).
Measured effluent/influent series -> interval-censored BV10; predictions from models trained on all
unflagged Koyama data; UV254 not reported -> imputed from groundwater SUVA in the training database."""
import numpy as np
import pandas as pd
from common import load, GBM, MLR, SAM, FEATS

from paths import RESULTS; OUT = str(RESULTS)
from paths import EPA_XLSX as EPA
# Haupert et al. 2023 (PMC10750846): F400, EBCT 3.45/3.58 min, pH 6.91/7.69, TOC 0.994/0.904 mg/L
TRAIN = {'preAS': dict(EBCT=3.45, pH=6.91, DOC=0.994), 'postAS': dict(EBCT=3.58, pH=7.69, DOC=0.904)}
F400 = dict(BET=933, pzc=8.29, mp_ratio=0.859796, PD=0, CD=0)


def read_curves(sheet):
    raw = pd.read_excel(EPA, sheet_name=sheet, header=None)
    comps = raw.iloc[1, 1:11].tolist()
    days = raw.iloc[3:, 0].astype(float).values
    inf = raw.iloc[3:, 1:11].astype(float).values
    eff = raw.iloc[3:, 11:21].astype(float).values
    return comps, days, inf, eff


def bv10_interval(days, inf, eff, ebct):
    """First sustained crossing of C/C0 >= 0.1 (two consecutive samples), C0 = campaign-mean influent.
    The day-0 sample is treated as start-up and ignored. Returns (lower, upper) in BV; upper=inf if right-censored."""
    bv = days * 1440.0 / ebct
    c0 = np.nanmean(inf)
    ratio = eff / c0
    idx = np.arange(1, len(days))
    for k in idx:
        nxt = ratio[k + 1] if k + 1 < len(days) else ratio[k]
        if ratio[k] >= 0.1 and nxt >= 0.1:
            lo = bv[k - 1] if k - 1 >= 1 else 0.0
            return lo, bv[k], c0
    return bv[-1], np.inf, c0


def main():
    d = load()
    ok = d[d.flag == 0]
    abr = d.drop_duplicates('Name of compound').set_index('Name of compound')[['E', 'S', 'A', 'B', 'V', 'L', 'charge']]
    gw = d[d.matrix_type.str.contains('ground', case=False, na=False) & (d.DOC > 0)]
    suva = (gw.UV254 * 100 / gw.DOC)
    suva = suva[suva > 0]
    suva_q = suva.quantile([0.1, 0.5, 0.9]).values
    rows = []
    for tname, sheet in [('preAS', 'Fig3_preAS'), ('postAS', 'Fig3_postAS')]:
        comps, days, inf, eff = read_curves(sheet)
        for j, c in enumerate(comps):
            lo, hi, c0 = bv10_interval(days, inf[:, j], eff[:, j], TRAIN[tname]['EBCT'])
            r = dict(train=tname, compound=c, C0=c0 / 1000.0, BV10_lo=lo, BV10_hi=hi, **TRAIN[tname], **F400)
            r.update(abr.loc[c].to_dict())
            rows.append(r)
    E = pd.DataFrame(rows)
    models = {n: M().fit(ok) for n, M in [('GBM', GBM), ('MLR', MLR), ('SAM', SAM)]}
    for k, sv in zip(['lo', 'mid', 'hi'], suva_q):
        E['UV254'] = sv * E['DOC'] / 100
        for n, m in models.items():
            E[f'pred_{n}_{k}'] = m.predict(E)
    E['UV254'] = suva_q[1] * E['DOC'] / 100
    E['ylo'] = np.log10(E.BV10_lo.clip(lower=1))
    E['yhi'] = np.log10(E.BV10_hi)
    E.attrs['suva'] = suva_q
    E.to_csv(f"{OUT}/epa_external.csv", index=False)
    print("SUVA quantiles used:", suva_q)
    print(E[['train', 'compound', 'C0', 'BV10_lo', 'BV10_hi', 'pred_GBM_mid', 'pred_SAM_mid', 'pred_MLR_mid']].round(3).to_string())


if __name__ == '__main__':
    main()
