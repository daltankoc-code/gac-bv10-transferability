import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paths import RESULTS; OUT = str(RESULTS)
R=pd.read_csv(f"{OUT}/fig1_lines_data.csv")
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
C={'GBM':'#2a6f97','SAM':'#c8553d','MLR':'#7a7a7a'}; M={'GBM':'o','SAM':'s','MLR':'^'}
sch=['original','random_point','LOSO','LOCO']
lab=['Original\nsplit','Point-level\n9:1','Leave-one-\nstudy-out','Leave-one-\ncampaign-out']
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
for i,m in enumerate(['GBM','SAM','MLR']):
    t=R[R.model==m].set_index('scheme').loc[sch]; x=np.arange(4)+(i-1)*0.09
    for a,(v,lo,hi,f) in zip(ax,[('mae','mae_lo','mae_hi',1),('o2','o2_lo','o2_hi',100)]):
        y=f*t[v].values; el=f*(t[v]-t[lo]).values; eh=f*(t[hi]-t[v]).values
        a.errorbar(x,y,yerr=[np.nan_to_num(el),np.nan_to_num(eh)],color=C[m],marker=M[m],ms=4.5,lw=1.3,capsize=2.5,elinewidth=0.9,label=m)
for a,yl in zip(ax,['Study-weighted MAE (log$_{10}$ BV$_{10}$)','Over-prediction > 2× (% of points)']):
    a.set_xticks(range(4)); a.set_xticklabels(lab,fontsize=7.5); a.set_ylabel(yl); a.set_xlim(-0.35,3.35); a.set_ylim(bottom=0)
    a.axvspan(1.5,3.35,color='0.94',zorder=0)
ax[0].legend(frameon=False,loc='upper left')
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_validation_schemes.png",dpi=300); fig.savefig(f"{OUT}/fig1_validation_schemes.pdf")
