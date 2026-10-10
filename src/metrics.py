"""Evaluation metrics with patient-level (cluster) bootstrap confidence intervals.

Several encounters can belong to one patient, so resampling encounters independently
would understate the variance. Every bootstrap replicate therefore resamples
*patients* with replacement and keeps all of each sampled patient's encounters.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

EPS = 1e-6


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------
def auroc(y, p):
    """Rank-based (Mann-Whitney) AUROC; ties get half credit."""
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auprc(y, p):
    return average_precision_score(y, p) if 0 < np.sum(y) < len(y) else np.nan


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _loglik(eta, y):
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def calibration_intercept_slope(y, p, tol=1e-8, max_iter=100):
    """Logistic recalibration of the linear predictor lp = logit(p):
    slope b from logit P(y=1) = a + b*lp, and calibration-in-the-large a from
    logit P(y=1) = a + offset(lp). Ideal values: intercept 0, slope 1.
    Damped Newton-Raphson (step halving) from the ideal (0, 1): the log-likelihood is
    concave, so every accepted step improves it and badly miscalibrated predictions
    cannot make the iteration diverge."""
    y = np.asarray(y, dtype=float)
    if y.min() == y.max():
        return np.nan, np.nan
    lp = logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))
    a, b = 0.0, 1.0
    ll = _loglik(a + b * lp, y)
    for _ in range(max_iter):
        q = expit(a + b * lp)
        w = q * (1 - q)
        r = y - q
        g0, g1 = r.sum(), (r * lp).sum()
        wl = w * lp
        h00, h01, h11 = w.sum(), wl.sum(), (wl * lp).sum()
        det = h00 * h11 - h01 * h01
        if not np.isfinite(det) or det <= 1e-12 * max(h00 * h11, 1e-300):
            return np.nan, np.nan
        da, db = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
        step = 1.0
        while True:
            a_new, b_new = a + step * da, b + step * db
            ll_new = _loglik(a_new + b_new * lp, y)
            if ll_new >= ll - 1e-10 or step < 1e-8:
                break
            step /= 2
        a, b, ll = a_new, b_new, ll_new
        if max(abs(step * da), abs(step * db)) < tol:
            break
    c = 0.0
    ll = _loglik(c + lp, y)
    for _ in range(max_iter):
        q = expit(c + lp)
        d = (y - q).sum() / max((q * (1 - q)).sum(), 1e-300)
        step = 1.0
        while True:
            ll_new = _loglik(c + step * d + lp, y)
            if ll_new >= ll - 1e-10 or step < 1e-8:
                break
            step /= 2
        c, ll = c + step * d, ll_new
        if abs(step * d) < tol:
            break
    return float(c), float(b)


def threshold_metrics(y, p, thr):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p) >= thr
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y)); tn = int(np.sum(~pred & ~y))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "f1": f1,
            "fpr": 1 - spec if spec == spec else np.nan, "flag_rate": float(pred.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def net_benefit(y, p, thresholds):
    """Net benefit of treating patients with p >= t: TP/n - FP/n * t/(1-t)."""
    y = np.asarray(y).astype(bool); p = np.asarray(p, dtype=float)
    n = len(y)
    t = np.asarray(thresholds, dtype=float)
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    k = np.searchsorted(-ps, -t, side="right")  # number of cases with p >= t
    ctp = np.concatenate([[0], np.cumsum(ys)])
    tp = ctp[k]; fp = k - tp
    w = t / (1 - t)
    prev = y.mean()
    return pd.DataFrame({"threshold": t, "nb_model": tp / n - fp / n * w,
                         "nb_all": prev - (1 - prev) * w, "nb_none": 0.0})


def reliability_curve(y, p, n_bins=10, strategy="quantile"):
    """Observed vs mean predicted risk per bin (quantile bins by default)."""
    p = np.asarray(p); y = np.asarray(y)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        k, n = y[m].sum(), m.sum()
        # Wilson 95% interval for the observed proportion
        z = 1.96; ph = k / n
        den = 1 + z * z / n
        centre = (ph + z * z / (2 * n)) / den
        half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        rows.append({"bin": b, "n": int(n), "mean_pred": float(p[m].mean()),
                     "observed": float(ph), "obs_lo": centre - half, "obs_hi": centre + half})
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins=10):
    rc = reliability_curve(y, p, n_bins)
    return float(np.sum(rc.n * np.abs(rc.observed - rc.mean_pred)) / rc.n.sum())


def point_metrics(y, p, thr):
    ci, cs = calibration_intercept_slope(y, p)
    out = {"n": int(len(y)), "events": int(np.sum(y)), "prevalence": float(np.mean(y)),
           "mean_pred": float(np.mean(p)), "auroc": auroc(y, p), "auprc": auprc(y, p),
           "brier": brier(y, p), "cal_intercept": ci, "cal_slope": cs,
           "oe_ratio": float(np.sum(y) / np.sum(p)), "ece": expected_calibration_error(y, p)}
    out.update(threshold_metrics(y, p, thr))
    return out


# ---------------------------------------------------------------------------
# patient-level cluster bootstrap
# ---------------------------------------------------------------------------
class ClusterBootstrap:
    """Pre-computes, for each replicate, the row indices of a patient-level resample.

    The same replicates are reused for every model on a given test set, so model
    comparisons (differences) are paired.
    """

    def __init__(self, groups, n_boot, seed):
        groups = pd.Series(pd.factorize(pd.Series(groups))[0])
        order = np.argsort(groups.to_numpy(), kind="stable")
        sorted_g = groups.to_numpy()[order]
        n_groups = sorted_g.max() + 1
        starts = np.searchsorted(sorted_g, np.arange(n_groups))
        sizes = np.bincount(sorted_g, minlength=n_groups)
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n_boot):
            g = rng.integers(0, n_groups, n_groups)
            sz = sizes[g]
            # rows of each sampled patient: start + 0..size-1
            offs = np.arange(sz.sum()) - np.repeat(np.cumsum(sz) - sz, sz)
            self.samples.append(order[np.repeat(starts[g], sz) + offs])

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


def ci(values, alpha=0.05):
    v = np.asarray([x for x in values if x == x])  # drop NaN
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


BOOT_KEYS = ["auroc", "auprc", "brier", "cal_intercept", "cal_slope", "oe_ratio",
             "sensitivity", "specificity", "ppv", "npv", "f1", "flag_rate"]


def bootstrap_metrics(y, p, thr, boot, fast_calibration=True):
    """Point estimates + percentile CIs for the main metrics."""
    y = np.asarray(y); p = np.asarray(p)
    point = point_metrics(y, p, thr)
    reps = {k: [] for k in BOOT_KEYS}
    for idx in boot:
        yb, pb = y[idx], p[idx]
        reps["auroc"].append(auroc(yb, pb))
        reps["auprc"].append(auprc(yb, pb))
        reps["brier"].append(brier(yb, pb))
        if fast_calibration:
            a, b = calibration_intercept_slope(yb, pb)
            reps["cal_intercept"].append(a); reps["cal_slope"].append(b)
        reps["oe_ratio"].append(yb.sum() / pb.sum())
        tm = threshold_metrics(yb, pb, thr)
        for k in ["sensitivity", "specificity", "ppv", "npv", "f1", "flag_rate"]:
            reps[k].append(tm[k])
    out = dict(point)
    for k, v in reps.items():
        if v:
            out[k + "_lo"], out[k + "_hi"] = ci(v)
    return out, reps


def bootstrap_net_benefit(y, p, thresholds, boot):
    y = np.asarray(y); p = np.asarray(p)
    base = net_benefit(y, p, thresholds)
    mats = np.array([net_benefit(y[i], p[i], thresholds).nb_model.to_numpy() for i in boot])
    base["nb_model_lo"] = np.quantile(mats, 0.025, axis=0)
    base["nb_model_hi"] = np.quantile(mats, 0.975, axis=0)
    return base
