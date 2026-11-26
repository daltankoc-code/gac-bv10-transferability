"""Summary of the GBM sensitivity to the number of terminal nodes J (Section 3.3).
Output: robust_gbm_leaves.csv."""
import numpy as np
import pandas as pd
from paths import RESULTS; OUT = str(RESULTS)

L = pd.read_csv(f"{OUT}/robust_predictions_GBM_leaves.csv")
B = pd.read_csv(f"{OUT}/predictions_GBM.csv").assign(J=4)
X = pd.concat([L, B[B.scheme.isin(['original', 'LOSO'])][L.columns]])
X = X[X.flag == 0].assign(e=lambda d: d.pred - d.y)
rows = []
for (J, sch), x in X.groupby(['J', 'scheme']):
    st = x.assign(a=x.e.abs()).groupby('study').a.mean()
    r2 = 1 - (x.e ** 2).sum() / ((x.y - x.y.mean()) ** 2).sum()
    trees = x.groupby('study' if sch == 'LOSO' else 'scheme').n_trees.first().median()
    rows.append(dict(J=J, scheme=sch, MAE_sw=st.mean(), R2=r2, over2x=(x.e > np.log10(2)).mean(),
                     bias=x.e.mean(), trees_median=trees))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/robust_gbm_leaves.csv", index=False)
print(R.round(3).to_string(index=False))
