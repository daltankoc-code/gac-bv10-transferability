import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paths import RESULTS; OUT = str(RESULTS)
P=pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")])
P=P[P.flag==0].assign(ae=lambda d:(d.pred-d.y).abs(), o2=lambda d:((d.pred-d.y)>np.log10(2)).astype(float))
def sw(d):  # study-weighted MAE and P(>2x)
    g=d.groupby('study')[['ae','o2']].mean(); return g.ae.mean(), g.o2.mean()
rng=np.random.default_rng(2024); B=5000; rows=[]
for m in ['GBM','SAM','MLR']:
    for sc in ['original','random_point','LOSO','LOCO']:
        d=P[(P.model==m)&(P.scheme==sc)]
        if sc=='random_point':
            v=np.array([sw(x) for _,x in d.groupby('rep')]); est=v.mean(0); lo=hi=[np.nan]*2
        else:
            est=np.array(sw(d)); lo=hi=[np.nan]*2
            if sc in ('LOSO','LOCO'):
                key='study' if sc=='LOSO' else 'campaign'
                per=d.groupby(['study','campaign'])[['ae','o2']].mean().reset_index()
                groups=per[key].unique(); bs=[]
                for _ in range(B):
                    pick=rng.choice(groups,len(groups),replace=True)
                    q=pd.concat([per[per[key]==k] for k in pick]); bs.append(q[['ae','o2']].mean().values)
                bs=np.array(bs); lo=np.percentile(bs,2.5,0); hi=np.percentile(bs,97.5,0)
        rows.append(dict(model=m,scheme=sc,mae=est[0],mae_lo=lo[0],mae_hi=hi[0],o2=est[1],o2_lo=lo[1],o2_hi=hi[1]))
R=pd.DataFrame(rows); R.to_csv(f"{OUT}/fig1_lines_data.csv",index=False); print(R.round(3).to_string())
