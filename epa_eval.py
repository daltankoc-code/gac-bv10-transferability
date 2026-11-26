import numpy as np, pandas as pd, glob
from paths import RESULTS; OUT = str(RESULTS)
def wq(r,g,t):
    w=1.0/pd.Series(g).map(pd.Series(g).value_counts()).values; o=np.argsort(r); cw=np.cumsum(w[o])/w.sum(); return r[o][np.searchsorted(cw,t)]
P=pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")])
E=pd.read_csv(f"{OUT}/epa_external.csv")
rows=[]
for m in ['GBM','SAM','MLR']:
    L=P[(P.scheme=='LOSO')&(P.flag==0)&(P.model==m)]
    r=(L.y-L.pred).values; g=L.study.values
    pm=E[f'pred_{m}_mid']
    d=dict(model=m, inside=((pm>=E.ylo)&(pm<=E.yhi)).mean(), above_upper=(pm>E.yhi).mean(),
           median_min_overfactor=10**np.median(pm-E.yhi), max_SUVA_effect=float((E[f'pred_{m}_hi']-E[f'pred_{m}_lo']).abs().max()))
    for t in [0.1,0.025]:
        q=wq(r,g,t); b=pm+q
        d[f'shift_{t}']=round(-q,3); d[f'early_definite_{t}']=(b>E.yhi).mean()
    rows.append(d)
R=pd.DataFrame(rows); R.to_csv(f"{OUT}/table_epa_summary.csv",index=False)
print(R.round(3).to_string(index=False))
