"""Step 5 - evaluation with patient-level bootstrap 95% CIs.

Produces (all under results/):
  metrics_overall.csv        every (model, test set): discrimination, calibration,
                             threshold metrics with CIs
  metrics_differences.csv    paired HGB-LR differences; internal-external drops
  dca.csv                    decision-curve net benefit (+ CI) vs treat-all / treat-none
  calibration_curves.csv     reliability-curve bins
  fairness_groups.csv        per-subgroup metrics with CIs
  fairness_gaps.csv          AUROC gap, equal-opportunity and equalized-odds differences
  fairness_calibration_curves.csv
  recalibration.json         sensitivity (c): recalibration fitted on Synthea half A
  posthoc_external_top_risk.csv  POST HOC: composition of the highest external risks
Usage:  python src/evaluate.py
"""
import json
import zlib

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import config as C
import harmonize as H
import metrics as M
from models import grouped_split

ORDERS = {"race": H.RACE_ORDER, "sex": H.SEX_ORDER, "age_band": H.AGE_BAND_ORDER,
          "insurance": H.INSURANCE_ORDER}
PRIMARY = ["d130|primary|lr", "d130|primary|hgb"]
FAIRNESS_SETS = ["d130_test", "synthea_all"]


def wide_frames(preds):
    """One frame per test set with a probability column per model (rows aligned)."""
    wide = {}
    for t, g in preds.groupby("test_set", sort=False):
        base = None
        for mid, gm in g.groupby("model_id", sort=False):
            gm = gm.reset_index(drop=True)
            if base is None:
                base = gm.drop(columns=["p", "model_id", "test_set"]).copy()
            else:
                assert (base.encounter_id.to_numpy() == gm.encounter_id.to_numpy()).all()
            base[mid] = gm.p.to_numpy()
        wide[t] = base
    return wide


def recalibrate(frame, models, seed):
    """Sensitivity (c): logistic recalibration fitted on a grouped half of the external
    cohort (A) and evaluated on the other half (B)."""
    a_idx, b_idx = grouped_split(frame.y, frame.patient_id, 2, seed)
    A, B = frame.iloc[a_idx], frame.iloc[b_idx].reset_index(drop=True)
    params = {}
    for mid in models:
        lpa = logit(np.clip(A[mid].to_numpy(), M.EPS, 1 - M.EPS))
        lpb = logit(np.clip(B[mid].to_numpy(), M.EPS, 1 - M.EPS))
        ci, cs = M.calibration_intercept_slope(A.y.to_numpy(), A[mid].to_numpy())
        # intercept+slope refit: a2 + b2*lp (calibration_intercept_slope returns the
        # slope of that model; its intercept is refitted here with the slope fixed)
        a2 = 0.0
        for _ in range(50):
            q = expit(a2 + cs * lpa)
            step = (A.y.to_numpy() - q).sum() / (q * (1 - q)).sum()
            a2 += step
            if abs(step) < 1e-10:
                break
        B[mid + "|recal_intercept"] = expit(ci + lpb)
        B[mid + "|recal_logistic"] = expit(a2 + cs * lpb)
        params[mid] = {"intercept_only": {"a": ci}, "logistic": {"a": a2, "b": cs},
                       "n_half_A": len(A), "n_half_B": len(B)}
    return B, params


def overall(frame, test_set, model_cols, thresholds, boot, reps_store):
    rows = []
    for mid in model_cols:
        base = mid.split("|recal_")[0]
        thr = thresholds[base]
        res, reps = M.bootstrap_metrics(frame.y.to_numpy(), frame[mid].to_numpy(), thr, boot)
        reps_store[(mid, test_set)] = reps
        rows.append({"model_id": mid, "test_set": test_set, "threshold": thr, **res})
    return rows


def spread(values_by_group, groups, key):
    """max - min of a metric across groups (NaNs ignored)."""
    v = np.array([values_by_group[g][key] for g in groups], dtype=float)
    return float(np.nanmax(v) - np.nanmin(v))


def fairness(frame, mid, test_set, thr, boot):
    y = frame.y.to_numpy(); p = frame[mid].to_numpy(); pred = p >= thr
    group_rows, gap_rows, curve_rows = [], [], []
    for attr in H.FAIRNESS_ATTRIBUTES:
        vals = frame[attr].to_numpy()
        groups = [g for g in ORDERS[attr] if (vals == g).any()]
        rep = []
        for g in groups:
            m = vals == g
            n, ev = int(m.sum()), int(y[m].sum())
            ok = n >= C.MIN_SUBGROUP_N and ev >= C.MIN_SUBGROUP_EVENTS \
                and (n - ev) >= C.MIN_SUBGROUP_EVENTS
            if ok:
                rep.append(g)
            tm = M.threshold_metrics(y[m], p[m], thr)
            ci_, cs_ = M.calibration_intercept_slope(y[m], p[m]) if ok else (np.nan, np.nan)
            group_rows.append({
                "model_id": mid, "test_set": test_set, "attribute": attr, "group": g,
                "n": n, "events": ev, "prevalence": ev / n if n else np.nan,
                "mean_pred": float(p[m].mean()), "reportable": ok,
                "auroc": M.auroc(y[m], p[m]) if ok else np.nan,
                "brier": M.brier(y[m], p[m]), "oe_ratio": ev / p[m].sum(),
                "cal_intercept": ci_, "cal_slope": cs_,
                "tpr": tm["sensitivity"], "fpr": tm["fpr"], "ppv": tm["ppv"],
                "npv": tm["npv"], "flag_rate": tm["flag_rate"]})
            if ok:
                rc = M.reliability_curve(y[m], p[m], n_bins=5)
                rc.insert(0, "group", g); rc.insert(0, "attribute", attr)
                rc.insert(0, "test_set", test_set); rc.insert(0, "model_id", mid)
                curve_rows.append(rc)
        # bootstrap: group metrics and gaps over the reportable groups
        stats = {g: {k: [] for k in ["auroc", "tpr", "fpr", "oe_ratio", "ppv"]} for g in rep}
        gaps = {k: [] for k in ["auroc_gap", "tpr_gap", "fpr_gap", "eo_diff", "oe_gap",
                                "ppv_gap"]}
        for idx in boot:
            yb, pb, db, vb = y[idx], p[idx], pred[idx], vals[idx]
            cur = {}
            for g in rep:
                m = vb == g
                yg, pg, dg = yb[m], pb[m], db[m]
                pos = yg.sum(); neg = len(yg) - pos
                cur[g] = {"auroc": M.auroc(yg, pg),
                          "tpr": (dg & (yg == 1)).sum() / pos if pos else np.nan,
                          "fpr": (dg & (yg == 0)).sum() / neg if neg else np.nan,
                          "oe_ratio": pos / pg.sum(),
                          "ppv": (dg & (yg == 1)).sum() / dg.sum() if dg.sum() else np.nan}
                for k, v in cur[g].items():
                    stats[g][k].append(v)
            if len(rep) >= 2:
                tg, fg = spread(cur, rep, "tpr"), spread(cur, rep, "fpr")
                gaps["auroc_gap"].append(spread(cur, rep, "auroc"))
                gaps["tpr_gap"].append(tg); gaps["fpr_gap"].append(fg)
                gaps["eo_diff"].append(max(tg, fg))
                gaps["oe_gap"].append(spread(cur, rep, "oe_ratio"))
                gaps["ppv_gap"].append(spread(cur, rep, "ppv"))
        for r in group_rows:
            if r["attribute"] == attr and r["group"] in stats and r["model_id"] == mid \
                    and r["test_set"] == test_set:
                for k, v in stats[r["group"]].items():
                    r[k + "_lo"], r[k + "_hi"] = M.ci(v)
        if len(rep) >= 2:
            pt = {r["group"]: r for r in group_rows if r["attribute"] == attr
                  and r["model_id"] == mid and r["test_set"] == test_set and r["reportable"]}
            row = {"model_id": mid, "test_set": test_set, "attribute": attr,
                   "groups_compared": ", ".join(rep),
                   "groups_not_reportable": ", ".join(g for g in groups if g not in rep),
                   **{k + "_gap": spread(pt, rep, m) for k, m in
                      [("auroc", "auroc"), ("tpr", "tpr"), ("fpr", "fpr"),
                       ("oe", "oe_ratio"), ("ppv", "ppv")]}}
            row["eo_diff"] = max(row["tpr_gap"], row["fpr_gap"])
            for k, v in gaps.items():
                row[k + "_lo"], row[k + "_hi"] = M.ci(v)
            # which groups define the extremes
            for k in ["auroc", "tpr", "fpr"]:
                vals_k = {g: pt[g][k] for g in rep}
                row[k + "_min_group"] = min(vals_k, key=vals_k.get)
                row[k + "_max_group"] = max(vals_k, key=vals_k.get)
            gap_rows.append(row)
    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    return group_rows, gap_rows, curves


def top_risk_decomposition(preds):
    """POST HOC (not pre-specified): where does the external signal come from?
    For the diabetes-130 models on the external cohort, compare the highest-risk
    fractions with the rest, and count how many of their readmissions were planned care
    (the next stay was chemotherapy / rehab / transplant) and how many carry a cancer
    flag."""
    syn = pd.read_parquet(C.DATA_PROCESSED / "synthea.parquet")[["encounter_id", "cm_cancer"]]
    rows = []
    for mid in PRIMARY:
        g = preds[(preds.model_id == mid) & (preds.test_set == "synthea_all")].merge(
            syn, on="encounter_id").sort_values("p", ascending=False).reset_index(drop=True)
        planned_next = (g.y == 1) & (g.y_unplanned == 0)
        for frac in [0.01, 0.02, 0.05, 0.10]:
            k = int(round(frac * len(g)))
            top, rest = g.iloc[:k], g.iloc[k:]
            rows.append({"model_id": mid, "top_fraction": frac, "n_top": k,
                         "readmit_rate_top": float(top.y.mean()),
                         "readmissions_top": int(top.y.sum()),
                         "planned_readmissions_top": int(planned_next.iloc[:k].sum()),
                         "cancer_share_top": float(top.cm_cancer.mean()),
                         "readmit_rate_rest": float(rest.y.mean()),
                         "cancer_share_rest": float(rest.cm_cancer.mean()),
                         "readmissions_all": int(g.y.sum()),
                         "planned_readmissions_all": int(planned_next.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "posthoc_external_top_risk.csv", index=False)
    return out


def main():
    preds = pd.read_parquet(C.RESULTS / "predictions.parquet")
    reg = json.loads((C.RESULTS / "models.json").read_text())
    thresholds = {k: v["threshold"] for k, v in reg.items()}
    wide = wide_frames(preds)

    # sensitivity (b): drop stays followed by death within 30 days without readmission
    sa = wide["synthea_all"]
    # sensitivity (f): unplanned outcome (first subsequent admission must be unplanned)
    wide["synthea_all_unplanned_outcome"] = sa.assign(y=sa.y_unplanned.astype(int))
    wide["synthea_all_excl_deaths"] = sa[~(sa.died_30d_after & (sa.y == 0))] \
        .reset_index(drop=True)
    # sensitivity (c): recalibration on half A, evaluation on half B
    d130_models_on_syn = [m for m in PRIMARY + ["d130|core|lr", "d130|core|hgb"]
                          if m in sa.columns]
    wide["synthea_half_B"], recal = recalibrate(sa, d130_models_on_syn, C.SEED)
    (C.RESULTS / "recalibration.json").write_text(json.dumps(recal, indent=2))

    boots, meta = {}, {}
    for t, fr in wide.items():
        # independent replicates per test set (stable per-name seed); paired across models
        boots[t] = M.ClusterBootstrap(fr.patient_id, C.N_BOOTSTRAP,
                                      C.SEED + zlib.crc32(t.encode()) % 10_000)
        meta[t] = {"rows": len(fr), "patients": int(fr.patient_id.nunique()),
                   "events": int(fr.y.sum()), "replicates": C.N_BOOTSTRAP}
    (C.RESULTS / "bootstrap_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- overall metrics -------------------------------------------------
    reps_store, rows = {}, []
    base_cols = {"encounter_id", "patient_id", "y", "race", "sex", "age_band", "insurance",
                 "died_30d_after", "y_unplanned", "planned"}
    for t, fr in wide.items():
        model_cols = [c for c in fr.columns if c not in base_cols]
        print(f"[eval] {t}: {len(fr):,} rows, models={model_cols}", flush=True)
        rows += overall(fr, t, model_cols, thresholds, boots[t], reps_store)
    overall_df = pd.DataFrame(rows)
    overall_df.to_csv(C.RESULTS / "metrics_overall.csv", index=False)

    # ---- differences -----------------------------------------------------
    diffs = []
    for t in wide:  # paired (same replicates): HGB - LR
        for src_fs in ["d130|primary", "d130|core", "synthea|primary", "synthea|core",
                       "d130|full"]:
            a, b = (src_fs + "|hgb", t), (src_fs + "|lr", t)
            if a in reps_store and b in reps_store:
                for k in ["auroc", "auprc", "brier"]:
                    d = np.array(reps_store[a][k]) - np.array(reps_store[b][k])
                    pa = overall_df.set_index(["model_id", "test_set"]).loc[a, k]
                    pb = overall_df.set_index(["model_id", "test_set"]).loc[b, k]
                    lo, hi = M.ci(d)
                    diffs.append({"comparison": "HGB - LR (paired)", "model": src_fs,
                                  "test_set": t, "metric": k, "estimate": pa - pb,
                                  "lo": lo, "hi": hi})
    for mid in PRIMARY + ["d130|core|lr", "d130|core|hgb"]:  # internal - external
        for ext in ["synthea_all", "synthea_test"]:
            a, b = (mid, "d130_test"), (mid, ext)
            if a in reps_store and b in reps_store:
                for k in ["auroc", "auprc", "brier"]:
                    d = np.array(reps_store[a][k]) - np.array(reps_store[b][k])
                    ix = overall_df.set_index(["model_id", "test_set"])
                    lo, hi = M.ci(d)
                    diffs.append({"comparison": f"internal - external ({ext})", "model": mid,
                                  "test_set": f"d130_test vs {ext}", "metric": k,
                                  "estimate": ix.loc[a, k] - ix.loc[b, k], "lo": lo, "hi": hi})
    pd.DataFrame(diffs).to_csv(C.RESULTS / "metrics_differences.csv", index=False)

    # ---- decision curves and calibration curves --------------------------
    dca_rows, cal_rows = [], []
    for t, fr in wide.items():
        model_cols = [c for c in fr.columns if c not in base_cols]
        for mid in model_cols:
            nb = M.bootstrap_net_benefit(fr.y.to_numpy(), fr[mid].to_numpy(),
                                         C.DCA_THRESHOLDS, boots[t])
            nb.insert(0, "test_set", t); nb.insert(0, "model_id", mid)
            dca_rows.append(nb)
            rc = M.reliability_curve(fr.y.to_numpy(), fr[mid].to_numpy(), n_bins=10)
            rc.insert(0, "test_set", t); rc.insert(0, "model_id", mid)
            cal_rows.append(rc)
    pd.concat(dca_rows).to_csv(C.RESULTS / "dca.csv", index=False)
    pd.concat(cal_rows).to_csv(C.RESULTS / "calibration_curves.csv", index=False)

    # ---- fairness ----------------------------------------------------------
    g_all, gap_all, curves_all = [], [], []
    for t in FAIRNESS_SETS:
        for mid in PRIMARY:
            print(f"[eval] fairness {mid} on {t}", flush=True)
            g, gp, cv = fairness(wide[t], mid, t, thresholds[mid], boots[t])
            g_all += g; gap_all += gp; curves_all.append(cv)
    pd.DataFrame(g_all).to_csv(C.RESULTS / "fairness_groups.csv", index=False)
    pd.DataFrame(gap_all).to_csv(C.RESULTS / "fairness_gaps.csv", index=False)
    pd.concat(curves_all).to_csv(C.RESULTS / "fairness_calibration_curves.csv", index=False)
    top_risk_decomposition(preds)
    print("[eval] done")


if __name__ == "__main__":
    main()
