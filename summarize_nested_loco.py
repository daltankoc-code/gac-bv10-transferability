import numpy as np, pandas as pd
from paths import RESULTS; OUT = str(RESULTS)
R=pd.read_csv(f"{OUT}/nested_loco_bounds.csv")
def ev(b,y,g):
    early=(y<b).astype(float); unused=np.clip(1-10**(b-y),0,None)
    df=pd.DataFrame({'g':g,'e':early,'u':unused}).groupby('g').mean()
    return df.e.mean(), df.u.mean(), (df.e>0.25).sum()
rows=[]
for m in ['GBM','SAM']:
    for unit,lab in [('campaign','inner LOCO (campaign-weighted)'),('study','inner LOSO (study-weighted)')]:
        for t in [0.05,0.10,0.20]:
            b=R[f'b_{m}_{unit}_{t}'].values
            ec,uc,nc=ev(b,R.y.values,R.campaign.values); es,us,_=ev(b,R.y.values,R.study.values)
            rows.append(dict(model=m,calibration=lab,tau=t,early_campaignw=ec,unused_campaignw=uc,early_studyw=es,unused_studyw=us,campaigns_early_gt25=nc))
T=pd.DataFrame(rows); T.to_csv(f"{OUT}/robust_nested_loco.csv",index=False); print(T.round(3).to_string(index=False))
for m in ['GBM','SAM']:
    sh=(R[f'pred_{m}']-R[f'b_{m}_campaign_0.1']).groupby(R.campaign).first()
    print(m,"inner-LOCO k0.10 range:",round(10**sh.min(),2),round(10**sh.median(),2),round(10**sh.max(),2))
