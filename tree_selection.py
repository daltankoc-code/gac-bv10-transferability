"""Table 4 of the article: GBM performance with the number of trees selected by random ten-fold CV
(original procedure, predictions_GBM.csv) or by study-grouped five-fold CV
(robust_predictions_GBM_groupedtrees.csv, produced by run_grouped_trees.py)."""
import numpy as np
import pandas as pd
from paths import RESULTS; OUT = str(RESULTS)

srcs = [('random-CV trees (as published)', 'predictions_GBM.csv'),
        ('study-grouped-CV trees', 'robust_predictions_GBM_groupedtrees.csv')]
rows = []
for name, f in srcs:
    P = pd.read_csv(f"{OUT}/{f}")
    P = P[P.flag == 0]
    for sch in ['original', 'LOSO', 'LOCO']:
        x = P[P.scheme == sch].assign(e=lambda d: d.pred - d.y)
        g = 'campaign' if sch == 'LOCO' else 'study'
        mae_gw = x.assign(a=x.e.abs()).groupby(g).a.mean().mean()
        r2 = 1 - (x.e ** 2).sum() / ((x.y - x.y.mean()) ** 2).sum()
        over2x = (x.e > np.log10(2)).mean()
        fold = 'study' if sch == 'LOSO' else ('campaign' if sch == 'LOCO' else 'scheme')
        trees = x.groupby(fold).n_trees.first().median()
        rows.append(dict(tree_selection=name, scheme=sch, MAE_groupw=mae_gw, R2=r2, over2x=over2x, trees_median=trees))
T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/robust_tree_selection.csv", index=False)
print(T.round(3).to_string(index=False))
