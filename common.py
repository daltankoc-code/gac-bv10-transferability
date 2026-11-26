"""Shared data loading, grouping and model definitions for the GAC BV10 transferability study."""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.model_selection import KFold, GroupKFold

from paths import KOYAMA_DB as DB
FEATS = ['C0', 'A', 'V', 'pH', 'mp_ratio', 'charge', 'BET', 'pzc', 'CD', 'PD',
         'EBCT', 'UV254', 'L', 'S', 'B', 'E', 'DOC']
CHEM = ['E', 'S', 'A', 'B', 'V', 'L', 'charge', 'C0']
MATRIX = ['DOC', 'UV254', 'pH']
COLUMN = ['BET', 'pzc', 'mp_ratio', 'EBCT', 'PD', 'CD']
SEED = 123


def load():
    d = pd.read_excel(DB, sheet_name="Data")
    d = d.rename(columns={'ref ID': 'study'})
    d['scale'] = np.where(d['scenario'].str.contains('pilot|full', case=False), 'pilot/full', 'RSSCT')
    d['campaign'] = campaign_groups(d)
    return d.reset_index(drop=True)


def _norm_matrix(s):
    s = str(s).strip()
    s = re.sub(r'^[A-D]\s*,\s*', '', s)          # "A, coag BEM" -> "coag BEM"
    s = s.lower().replace('coagulated', 'coag').replace('dilco ii', 'co ii')
    s = re.sub(r'\s+', ' ', s)
    if s in ('oh', 'oh gw'):
        s = 'oh'
    if s.startswith('d,') or s == 'lm':
        s = 'lm'
    return s


def campaign_groups(d):
    """Union of studies that share an identical (normalised) source water."""
    parent = {s: s for s in d['study'].unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    m = d.assign(nm=d['matrix name'].map(_norm_matrix))
    generic = {'surface water', 'synthetic water', 'waste water', 'secondary effluent', 'wwtp eff',
               'well water', 'conventional'}
    for nm, sub in m.groupby('nm'):
        if nm in generic:
            continue
        studies = sub['study'].unique()
        for s in studies[1:]:
            parent[find(s)] = find(studies[0])
    roots = d['study'].map(find)
    lab = {r: i for i, r in enumerate(sorted(roots.unique()))}
    return roots.map(lab)


# ---------------------------------------------------------------- models
GBM_PARAMS = dict(learning_rate=0.1, max_leaf_nodes=4, min_samples_leaf=10,
                  subsample=0.8, random_state=SEED)   # gbm(interaction.depth=3) == 4 terminal nodes
MAX_TREES = 3000


def gbm_select_trees(X, y, n_splits=10, max_trees=MAX_TREES, loss='squared_error', alpha=0.5):
    """Choose number of trees by random K-fold CV inside the training data (as in the original R script)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    err = np.zeros(max_trees)
    for tr, va in kf.split(X):
        m = GradientBoostingRegressor(n_estimators=max_trees, loss=loss, alpha=alpha, **GBM_PARAMS)
        m.fit(X[tr], y[tr])
        for k, p in enumerate(m.staged_predict(X[va])):
            if loss == 'quantile':
                r = y[va] - p
                err[k] += np.sum(np.maximum(alpha * r, (alpha - 1) * r))
            else:
                err[k] += np.sum((y[va] - p) ** 2)
    return int(np.argmin(err) + 1)


class GBM:
    name = 'GBM'

    def __init__(self, n_trees=None, loss='squared_error', alpha=0.5, inner_splits=10):
        self.n_trees, self.loss, self.alpha, self.inner = n_trees, loss, alpha, inner_splits

    def fit(self, D):
        X, y = D[FEATS].values, D['y'].values
        n = self.n_trees or gbm_select_trees(X, y, self.inner, loss=self.loss, alpha=self.alpha)
        self.n_used = n
        self.m = GradientBoostingRegressor(n_estimators=n, loss=self.loss, alpha=self.alpha, **GBM_PARAMS).fit(X, y)
        return self

    def predict(self, D):
        return self.m.predict(D[FEATS].values)


class MLR:
    """Ordinary least squares on standardised inputs, with C0, DOC, UV254 and EBCT log10-transformed
    as in the SAM (see _prep)."""
    name = 'MLR'

    def fit(self, D):
        Z = _prep(D).values
        self.sc = StandardScaler().fit(Z)
        self.m = LinearRegression().fit(self.sc.transform(Z), D['y'].values)
        return self

    def predict(self, D):
        return self.m.predict(self.sc.transform(_prep(D).values))


class MEAN:
    """Null predictor: mean of log10 BV10 in the training data."""
    name = 'MEAN'

    def fit(self, D):
        self.c = D['y'].mean()
        return self

    def predict(self, D):
        return np.full(len(D), self.c)


LOGVARS = ['C0', 'DOC', 'UV254', 'EBCT']
SPLINE_VARS = ['E', 'S', 'A', 'B', 'V', 'L', 'C0', 'DOC', 'UV254', 'pH', 'BET', 'pzc', 'mp_ratio', 'EBCT']
LINEAR_VARS = ['charge', 'PD', 'CD']
INTERACT = [('DOC', 'L'), ('DOC', 'A'), ('DOC', 'V'), ('DOC', 'charge'), ('UV254', 'L')]


def _prep(D):
    Z = D[FEATS].copy().astype(float)
    for v in LOGVARS:
        Z[v] = np.log10(Z[v].clip(lower=1e-4))
    return Z


class SAM:
    """Structured additive model: penalised cubic B-spline main effects for chemistry, matrix and
    column/GAC blocks plus pre-specified matrix x chemistry interactions; ridge penalty chosen by
    study-grouped inner CV on the training studies only."""
    name = 'SAM'
    INTER = INTERACT
    ALPHAS = np.logspace(-3, 2.5, 23)

    def _design(self, Z, fit=False):
        blocks = []
        if fit:
            self.spl = {v: SplineTransformer(n_knots=4, degree=3, knots='quantile', extrapolation='linear').fit(Z[[v]])
                        for v in SPLINE_VARS}
        for v in SPLINE_VARS:
            blocks.append(self.spl[v].transform(Z[[v]]))
        blocks.append(Z[LINEAR_VARS].values)
        if fit:
            self.zs = StandardScaler().fit(Z[FEATS].values)
        zz = pd.DataFrame(self.zs.transform(Z[FEATS].values), columns=FEATS, index=Z.index)
        if self.INTER:
            blocks.append(np.column_stack([zz[a] * zz[b] for a, b in self.INTER]))
        X = np.hstack(blocks)
        if fit:
            self.xs = StandardScaler().fit(X)
        return self.xs.transform(X)

    def fit(self, D):
        Z = _prep(D)
        y = D['y'].values
        g = D['study'].values
        ng = len(np.unique(g))
        gkf = GroupKFold(n_splits=min(5, ng))
        best, best_a = np.inf, None
        for a in self.ALPHAS:
            e = []
            for tr, va in gkf.split(Z, y, g):
                s = type(self).__new__(type(self))
                Xtr = s._design(Z.iloc[tr], fit=True)
                m = Ridge(alpha=a * len(tr)).fit(Xtr, y[tr])
                p = m.predict(s._design(Z.iloc[va]))
                # equal weight per held-out study
                dfv = pd.DataFrame({'g': g[va], 'ae': np.abs(y[va] - p)})
                e.append(dfv.groupby('g')['ae'].mean().values)
            e = np.mean(np.concatenate(e))
            if e < best:
                best, best_a = e, a
        self.alpha = best_a
        X = self._design(Z, fit=True)
        self.m = Ridge(alpha=best_a * len(y)).fit(X, y)
        return self

    def predict(self, D):
        return self.m.predict(self._design(_prep(D)))


class SAM_NOINT(SAM):
    """Ablation of the SAM without the pre-specified interaction terms (P empty)."""
    name = 'SAM_NOINT'
    INTER = []


MODELS = {'GBM': GBM, 'MLR': MLR, 'SAM': SAM, 'SAM_NOINT': SAM_NOINT, 'MEAN': MEAN}


# ---------------------------------------------------------------- metrics
def metrics(df, pred='pred', obs='y', group='study'):
    e = df[pred] - df[obs]
    per = df.assign(ae=e.abs(), se=e, over=(e > 0), over2=(e > np.log10(2))).groupby(group)
    ss_res = np.sum(e ** 2)
    ss_tot = np.sum((df[obs] - df[obs].mean()) ** 2)
    return dict(n=len(df), n_groups=df[group].nunique(),
                MAE=e.abs().mean(), MAE_studyw=per['ae'].mean().mean(),
                RMSE=np.sqrt(np.mean(e ** 2)), R2=1 - ss_res / ss_tot,
                bias=e.mean(), over_rate=(e > 0).mean(), over2x_rate=(e > np.log10(2)).mean(),
                over2x_studyw=per['over2'].mean().mean())
