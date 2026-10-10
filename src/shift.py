"""Step 5b - cohort description and dataset-shift diagnostics.

* Table 1 for both analysis cohorts with standardized mean differences (SMD);
  |SMD| > 0.1 is the usual threshold for a meaningful imbalance.
* Domain classifier: a gradient-boosting model trained to tell diabetes-130 rows from
  Synthea rows using the model features (grouped 5-fold CV by patient). An AUROC of
  0.5 means the two cohorts are indistinguishable; 1.0 means fully separable.

Outputs: results/table1.csv, results/shift_smd.csv, results/domain_classifier.json
Usage:  python src/shift.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import config as C
import harmonize as H

LABELS = {"age_years": "Age (band midpoint), years", "los_days": "Length of stay, days",
          "n_inpatient_prior": "Inpatient stays, prior year",
          "n_emergency_prior": "Emergency visits, prior year",
          "n_outpatient_prior": "Outpatient visits, prior year",
          "n_diagnoses": "Number of diagnoses (capped at 9)",
          "n_medications": "Number of medications", "female": "Female",
          **{k: v[0] for k, v in H.COMORBIDITIES.items()}}


def smd_cont(a, b):
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (b.mean() - a.mean()) / s if s > 0 else 0.0


def smd_prop(p1, p2):
    s = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p2 - p1) / s if s > 0 else 0.0


def main():
    d = pd.read_parquet(C.DATA_PROCESSED / "d130.parquet")
    s = pd.read_parquet(C.DATA_PROCESSED / "synthea.parquet")
    rows = [{"variable": "Encounters / stays", "d130": f"{len(d):,}", "synthea": f"{len(s):,}",
             "smd": np.nan},
            {"variable": "Patients", "d130": f"{d.patient_id.nunique():,}",
             "synthea": f"{s.patient_id.nunique():,}", "smd": np.nan},
            {"variable": "30-day readmission, %", "d130": f"{100 * d.y.mean():.1f}",
             "synthea": f"{100 * s.y.mean():.1f}",
             "smd": smd_prop(d.y.mean(), s.y.mean())}]
    smd_rows = []
    for f in H.NUMERIC_FEATURES:
        v = smd_cont(d[f], s[f])
        rows.append({"variable": LABELS[f] + ", mean (SD)",
                     "d130": f"{d[f].mean():.2f} ({d[f].std():.2f})",
                     "synthea": f"{s[f].mean():.2f} ({s[f].std():.2f})", "smd": v})
        smd_rows.append({"feature": f, "label": LABELS[f], "type": "numeric",
                         "d130": d[f].mean(), "synthea": s[f].mean(), "smd": v})
    for f in H.BINARY_FEATURES:
        v = smd_prop(d[f].mean(), s[f].mean())
        rows.append({"variable": LABELS[f] + ", %", "d130": f"{100 * d[f].mean():.1f}",
                     "synthea": f"{100 * s[f].mean():.1f}", "smd": v})
        smd_rows.append({"feature": f, "label": LABELS[f], "type": "binary",
                         "d130": d[f].mean(), "synthea": s[f].mean(), "smd": v})
    for attr, order in [("age_band", H.AGE_BAND_ORDER), ("race", H.RACE_ORDER),
                        ("insurance", H.INSURANCE_ORDER)]:
        for g in order:
            p1, p2 = (d[attr] == g).mean(), (s[attr] == g).mean()
            rows.append({"variable": f"{attr}: {g}, %", "d130": f"{100 * p1:.1f}",
                         "synthea": f"{100 * p2:.1f}", "smd": smd_prop(p1, p2)})
    t1 = pd.DataFrame(rows)
    t1["smd"] = t1.smd.round(3)
    t1.to_csv(C.RESULTS / "table1.csv", index=False)
    pd.DataFrame(smd_rows).to_csv(C.RESULTS / "shift_smd.csv", index=False)

    # ---- domain classifier ---------------------------------------------------
    out = {}
    both = pd.concat([d.assign(domain=0), s.assign(domain=1)], ignore_index=True)
    for name, feats in [("primary", H.FEATURES), ("core", H.CORE_FEATURES)]:
        oof = np.zeros(len(both))
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
        for tr, va in cv.split(both, both.domain, both.patient_id):
            m = HistGradientBoostingClassifier(random_state=C.SEED).fit(
                both[feats].iloc[tr], both.domain.iloc[tr])
            oof[va] = m.predict_proba(both[feats].iloc[va])[:, 1]
        out[name] = {"features": feats, "auroc": float(roc_auc_score(both.domain, oof))}
        # single-feature separability, to show which features drive the shift
        out[name]["per_feature_auroc"] = {
            f: float(max(roc_auc_score(both.domain, both[f]),
                         1 - roc_auc_score(both.domain, both[f]))) for f in feats}
    (C.RESULTS / "domain_classifier.json").write_text(json.dumps(out, indent=2))
    print(t1.to_string())
    print({k: round(v["auroc"], 4) for k, v in out.items()})


if __name__ == "__main__":
    main()
