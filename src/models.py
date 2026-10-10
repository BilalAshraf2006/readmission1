"""Model pipelines (logistic regression, histogram gradient boosting) and grouped CV.

Pre-processing lives inside each sklearn Pipeline, so it is re-fitted on every training
fold and cannot leak test-fold information.
"""
import itertools
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, \
    StandardScaler


class ClipToTrainRange(BaseEstimator, TransformerMixin):
    """Clip each numeric column to the [min, max] seen in training, so that a model
    never extrapolates beyond the training support (relevant for external data)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanmin(X, axis=0)
        self.hi_ = np.nanmax(X, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


LR_GRID = [{"C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0]]
HGB_GRID = [dict(zip(["max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"],
                     v))
            for v in itertools.product([200, 400], [15, 31], [20, 100, 400], [0.0, 1.0])]


def make_model(kind, spec, params, seed):
    """spec: dict with lists 'numeric', 'counts', 'binary', 'categorical'."""
    numeric = [c for c in spec["numeric"] if c not in spec["counts"]]
    counts, binary, cat = spec["counts"], spec["binary"], spec.get("categorical", [])
    if kind == "lr":
        tf = [("num", Pipeline([("clip", ClipToTrainRange()), ("sc", StandardScaler())]),
               numeric),
              ("cnt", Pipeline([("clip", ClipToTrainRange()),
                                ("log", FunctionTransformer(np.log1p)),
                                ("sc", StandardScaler())]), counts),
              ("bin", "passthrough", binary)]
        if cat:
            tf.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat))
        pre = ColumnTransformer([t for t in tf if t[2]], remainder="drop")
        clf = LogisticRegression(C=params["C"], max_iter=5000)
        return Pipeline([("pre", pre), ("clf", clf)])
    if kind == "hgb":
        tf = [("num", ClipToTrainRange(), numeric + counts), ("bin", "passthrough", binary)]
        if cat:
            tf.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1, encoded_missing_value=-1),
                       cat))
        tf = [t for t in tf if t[2]]
        pre = ColumnTransformer(tf, remainder="drop")
        n_before = len(numeric + counts) + len(binary)
        cat_mask = [False] * n_before + [True] * len(cat) if cat else None
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, early_stopping=False, random_state=seed,
            categorical_features=cat_mask, **params)
        return Pipeline([("pre", pre), ("clf", clf)])
    raise ValueError(kind)


def grouped_split(y, groups, n_folds, seed):
    """Return (train_idx, test_idx): one fold of a StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(np.zeros(len(y)), y, groups))
    return tr, te


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def _fit_predict(kind, spec, params, seed, X, y, tr, va):
    m = make_model(kind, spec, params, seed).fit(X.iloc[tr], y[tr])
    return m.predict_proba(X.iloc[va])[:, 1]


def tune(kind, spec, X, y, groups, n_folds, seed, log=print, n_jobs=None):
    """Grid search with StratifiedGroupKFold; selection by mean validation log loss.

    Returns best params, the full CV table, and the out-of-fold predictions of the best
    configuration (used to choose the decision threshold without touching test data).
    All (configuration, fold) fits run in parallel worker processes.
    """
    grid = LR_GRID if kind == "lr" else HGB_GRID
    folds = list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                 .split(X, y, groups))
    y = np.asarray(y)
    t0 = time.time()
    n_jobs = n_jobs or min(os.cpu_count() or 1, len(grid) * len(folds))
    preds = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(kind, spec, params, seed, X, y, tr, va)
        for params in grid for tr, va in folds)
    rows, oofs = [], []
    for i, params in enumerate(grid):
        oof = np.zeros(len(y))
        for j, (_, va) in enumerate(folds):
            oof[va] = preds[i * len(folds) + j]
        fold_ll = [log_loss(y[va], oof[va]) for _, va in folds]
        fold_auc = [roc_auc_score(y[va], oof[va]) for _, va in folds]
        rows.append({**params, "cv_logloss": np.mean(fold_ll), "cv_logloss_sd": np.std(fold_ll),
                     "cv_auroc": np.mean(fold_auc), "cv_auroc_sd": np.std(fold_auc)})
        oofs.append(oof)
    table = pd.DataFrame(rows)
    best = int(table.cv_logloss.idxmin())
    log(f"    {kind}: {len(grid)} configs x {len(folds)} folds in {time.time() - t0:.0f}s; "
        f"best logloss={table.cv_logloss[best]:.5f} auroc={table.cv_auroc[best]:.4f}")
    best_params = {k: grid[best][k] for k in grid[best]}
    return best_params, table, oofs[best]
