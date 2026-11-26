import numpy as np, pandas as pd, glob
from paths import RESULTS; OUT = str(RESULTS); rng=np.random.default_rng(2026); B=5000
P=pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")]); P=P[P.flag==0]
P['ae']=(P.pred-P.y).abs()
rows=[]
for m in ['GBM','SAM','MLR','MEAN']:
    rnd=P[(P.scheme=='random_point')&(P.model==m)]
    den=rnd.groupby(['rep','study']).ae.mean().groupby('rep').mean()
    for sch,unit in [('LOSO','study'),('LOCO','campaign')]:
        x=P[(P.scheme==sch)&(P.model==m)]
        st=x.groupby('study').agg(ae=('ae','mean'),camp=('campaign','first'))
        est=st.ae.mean()
        units=st.index.values if unit=='study' else st.camp.unique()
        bs=[];rs=[]
        for _ in range(B):
            pick=rng.choice(units,len(units))
            vals=np.concatenate([st.ae.values[(st.index.values if unit=='study' else st.camp.values)==u] for u in pick])
            v=vals.mean(); bs.append(v); rs.append(v/den.values[rng.integers(len(den))])
        rows.append(dict(model=m,scheme=sch,boot_unit=unit,MAE_sw=est,ci_lo=np.percentile(bs,2.5),ci_hi=np.percentile(bs,97.5),
                         MAE_sw_point=den.mean(),rho=est/den.mean(),rho_lo=np.percentile(rs,2.5),rho_hi=np.percentile(rs,97.5)))
T=pd.DataFrame(rows); T.to_csv(f"{OUT}/robust_ratio_studyweighted.csv",index=False); print(T.round(3).to_string(index=False))
