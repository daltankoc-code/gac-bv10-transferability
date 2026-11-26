import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import RESULTS; OUT = str(RESULTS)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
C = {'GBM': '#2a6f97', 'SAM': '#c8553d', 'MLR': '#7a7a7a'}

# Fig. 2 of the article: signed LOSO errors per withheld study (GBM and SAM)
P = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/predictions_*.csv")])
L = P[(P.scheme == 'LOSO') & (P.flag == 0)].assign(err=lambda x: x.pred - x.y)
fig, ax = plt.subplots(figsize=(7.2, 2.8))
studies = sorted(L.study.unique())
for j, m in enumerate(['GBM', 'SAM']):
    for k, s in enumerate(studies):
        e = L[(L.model == m) & (L.study == s)]['err']
        ax.scatter(np.full(len(e), k + (j - 0.5) * 0.3), e, s=6, color=C[m], alpha=0.6, label=m if k == 0 else None)
ax.axhline(0, color='k', lw=0.6); ax.axhline(np.log10(2), color='k', lw=0.6, ls='--')
ax.text(len(studies) - 0.5, np.log10(2) + 0.03, '2× over-prediction', ha='right', fontsize=7)
scale = L[L.model == 'GBM'].groupby('study')['scale'].agg(lambda s: 'P' if (s == 'pilot/full').all() else ('R' if (s == 'RSSCT').all() else 'M'))
ax.set_xticks(range(len(studies))); ax.set_xticklabels([f"{s}\n{scale[s]}" for s in studies], fontsize=7)
ax.set_ylabel('Predicted − observed\n(log$_{10}$ BV$_{10}$)'); ax.set_xlabel('Held-out study (P: pilot/full, R: RSSCT, M: mixed)')
ax.legend(frameon=False, loc='upper left', markerscale=2.5, ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_loso_per_study.png", dpi=300)

