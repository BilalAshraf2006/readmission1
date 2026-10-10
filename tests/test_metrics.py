import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import metrics as M


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    p = np.round(rng.random(3000) * 0.3 + 0.2 * y, 2)  # many ties
    assert np.isclose(M.auroc(y, p), roc_auc_score(y, p))


def test_calibration_recovers_known_values():
    rng = np.random.default_rng(1)
    lp = rng.normal(-2, 1, 200_000)
    y = (rng.random(lp.size) < 1 / (1 + np.exp(-lp))).astype(int)
    a, b = M.calibration_intercept_slope(y, 1 / (1 + np.exp(-lp)))
    assert abs(a) < 0.03 and abs(b - 1) < 0.03
    # predictions twice as extreme -> slope ~0.5
    _, b2 = M.calibration_intercept_slope(y, 1 / (1 + np.exp(-(2 * lp + 2))))
    assert abs(b2 - 0.5) < 0.03


def test_cluster_bootstrap_keeps_whole_patients():
    g = np.array(["a", "a", "b", "c", "c", "c"])
    for idx in M.ClusterBootstrap(g, 50, 3):
        s = pd.Series(g[idx]).value_counts()
        for grp, cnt in s.items():
            assert cnt % (g == grp).sum() == 0


def test_net_benefit_hand_computed():
    nb = M.net_benefit(np.array([1, 0, 0, 0]), np.array([.9, .6, .2, .1]), [0.1, 0.5])
    # t=0.1: all 4 treated -> 1/4 - 3/4 * (0.1/0.9)
    assert np.isclose(nb.nb_model[0], 0.25 - 0.75 / 9)
    # t=0.5: 0.9 and 0.6 treated -> 1/4 - 1/4 * 1
    assert np.isclose(nb.nb_model[1], 0.0)
    assert np.isclose(nb.nb_all[1], 0.25 - 0.75)


def test_threshold_metrics_confusion_matrix():
    tm = M.threshold_metrics([1, 1, 0, 0, 0], [0.9, 0.2, 0.8, 0.1, 0.3], 0.5)
    assert (tm["tp"], tm["fn"], tm["fp"], tm["tn"]) == (1, 1, 1, 2)
    assert np.isclose(tm["ppv"], 0.5) and np.isclose(tm["specificity"], 2 / 3)


def test_calibration_converges_for_badly_miscalibrated_predictions():
    """Regression test: undamped Newton from (0, 1) diverged on predictions with a
    large calibration-in-the-large offset and a flat slope."""
    from scipy.special import expit, logit
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(2)
    lp_true = rng.normal(-2, 0.4, 20_000)
    y = (rng.random(lp_true.size) < expit(lp_true)).astype(int)
    p = expit(3.5 * lp_true + 2.0)  # far too extreme and badly shifted
    a, b = M.calibration_intercept_slope(y, p)
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    ref = LogisticRegression(C=1e9, max_iter=5000).fit(lp[:, None], y).coef_[0, 0]
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(b - ref) < 1e-3
