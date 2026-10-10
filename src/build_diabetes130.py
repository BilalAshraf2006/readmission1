"""Step 3a - build the diabetes-130 analysis cohort.

* Label: readmitted == "<30" -> 1; "NO" and ">30" -> 0.
* Exclusions (each step is counted and written to results/cohort_flow_d130.json):
    1. discharge_disposition_id in {11, 13, 14, 19, 20, 21} (died or discharged to
       hospice; such patients cannot be readmitted);
    2. gender == "Unknown/Invalid" (sex is a model feature);
    3. planned-care index admissions: primary diagnosis V57 (rehabilitation) or V58
       (chemotherapy / other aftercare) - Amendment 1 (planned care).
* Missing values: the file codes missing values as "?". The literal string "None" in
  A1Cresult / max_glu_serum means "test not performed" and is kept as a category (by
  default pandas would silently turn it into NaN).
* Near-empty / unusable columns: weight (~97% missing) is dropped; payer_code (~40%
  missing) is not a model feature but is kept as the insurance audit attribute with an
  explicit "Unknown" group; medical_specialty (~49% missing) has no Synthea analogue, so
  it is excluded from the shared feature set and used (with an explicit "Missing"
  level) only in the internal full-feature sensitivity model.

Usage:  python src/build_diabetes130.py
"""
import json

import numpy as np
import pandas as pd

import config as C
import harmonize as H

PLANNED_DIAG1 = ["V57", "V58"]  # CMS "always planned": rehabilitation, chemotherapy
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]


def icd9_chapter(code):
    """Coarse ICD-9 group of the primary diagnosis (for the full-feature model only)."""
    s, cat = H.normalize_icd9(code)
    if s is None:
        return "Missing"
    if s.startswith("250"):
        return "Diabetes"
    if cat is None:
        return "V/E codes"
    for lo, hi, name in [(390, 459, "Circulatory"), (785, 785, "Circulatory"),
                         (460, 519, "Respiratory"), (786, 786, "Respiratory"),
                         (520, 579, "Digestive"), (787, 787, "Digestive"),
                         (800, 999, "Injury"), (710, 739, "Musculoskeletal"),
                         (580, 629, "Genitourinary"), (788, 788, "Genitourinary"),
                         (140, 239, "Neoplasms")]:
        if lo <= cat <= hi:
            return name
    return "Other"


def main():
    raw = pd.read_csv(C.D130_DIR / "diabetic_data.csv", keep_default_na=False,
                      na_values=["?"], low_memory=False)
    flow = [{"step": "All encounters in diabetic_data.csv", "encounters": len(raw),
             "patients": int(raw.patient_nbr.nunique()),
             "readmit_30d": int((raw.readmitted == "<30").sum())}]

    missing = (raw.isna().mean() * 100).round(2).sort_values(ascending=False)
    missing = missing[missing > 0].rename("pct_missing").to_frame()

    dead = raw.discharge_disposition_id.isin(C.D130_DEATH_HOSPICE_CODES)
    dropped_by_code = raw.loc[dead, "discharge_disposition_id"].value_counts().sort_index()
    df = raw.loc[~dead].copy()
    flow.append({"step": "Excluded: died or discharged to hospice "
                         "(discharge_disposition_id 11,13,14,19,20,21)",
                 "excluded": int(dead.sum()),
                 "excluded_by_code": {int(k): int(v) for k, v in dropped_by_code.items()},
                 "excluded_readmit_30d": int((raw.loc[dead, "readmitted"] == "<30").sum()),
                 "encounters": len(df), "patients": int(df.patient_nbr.nunique())})

    bad_sex = df.gender == "Unknown/Invalid"
    df = df.loc[~bad_sex].copy()
    flow.append({"step": "Excluded: gender 'Unknown/Invalid'", "excluded": int(bad_sex.sum()),
                 "encounters": len(df), "patients": int(df.patient_nbr.nunique()),
                 "readmit_30d": int((df.readmitted == "<30").sum())})

    planned = df.diag_1.isin(PLANNED_DIAG1)
    flow.append({"step": "Excluded: planned-care admission (primary diagnosis V57 "
                         "rehabilitation or V58 chemotherapy/aftercare)",
                 "excluded": int(planned.sum()),
                 "excluded_by_code": {k: int(v) for k, v in
                                      df.loc[planned, "diag_1"].value_counts().items()},
                 "excluded_readmit_30d": int((df.loc[planned, "readmitted"] == "<30").sum())})
    df = df.loc[~planned].copy()
    flow[-1].update({"encounters": len(df), "patients": int(df.patient_nbr.nunique()),
                     "readmit_30d": int((df.readmitted == "<30").sum())})

    out = pd.DataFrame(index=df.index)
    out["encounter_id"] = df.encounter_id.astype(str)
    out["patient_id"] = "d130_" + df.patient_nbr.astype(str)
    out["y"] = (df.readmitted == "<30").astype(np.int8)

    # --- shared features --------------------------------------------------
    age_lo = df.age.str.extract(r"\[(\d+)-")[0].astype(int)
    out["age_years"] = age_lo + 5.0  # band midpoint
    out["los_days"] = df.time_in_hospital.astype(float)
    out["n_inpatient_prior"] = df.number_inpatient.astype(float)
    out["n_emergency_prior"] = df.number_emergency.astype(float)
    out["n_outpatient_prior"] = df.number_outpatient.astype(float)
    # capped at 9: the recording limit for all but 0.11% of encounters (see README)
    out["n_diagnoses"] = df.number_diagnoses.clip(upper=9).astype(float)
    out["n_medications"] = df.num_medications.astype(float)
    out["female"] = (df.gender == "Female").astype(np.int8)
    out = out.join(H.icd9_flags(df))

    # --- audit attributes -------------------------------------------------
    out["race"] = df.race.fillna("Unknown")
    out["sex"] = df.gender
    out["age_band"] = H.age_band_from_years(out.age_years.to_numpy()).to_numpy()
    out["insurance"] = H.d130_insurance(df.payer_code)
    out["payer_code_raw"] = df.payer_code.fillna("?")

    # --- extra columns for the internal full-feature sensitivity model -----
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        out["x_" + c] = df[c].astype(str)
    out["x_num_lab_procedures"] = df.num_lab_procedures.astype(float)
    out["x_num_procedures"] = df.num_procedures.astype(float)
    out["x_A1Cresult"] = df.A1Cresult  # "None" = not measured (kept as a level)
    out["x_max_glu_serum"] = df.max_glu_serum
    for c in MED_COLS + ["change", "diabetesMed"]:
        out["x_" + c] = df[c]
    spec = df.medical_specialty.fillna("Missing")
    top = spec[spec != "Missing"].value_counts().index[:10]
    out["x_medical_specialty"] = spec.where(spec.isin(top) | (spec == "Missing"), "Other")
    out["x_diag1_group"] = df.diag_1.map(icd9_chapter)
    out["x_diag_missing_n"] = df[["diag_1", "diag_2", "diag_3"]].isna().sum(axis=1)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_parquet(C.DATA_PROCESSED / "d130.parquet", index=False)

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    flow.append({"step": "Analysis cohort", "encounters": len(out),
                 "patients": int(out.patient_id.nunique()),
                 "readmit_30d": int(out.y.sum()), "prevalence": round(float(out.y.mean()), 4),
                 "encounters_per_patient_max": int(out.patient_id.value_counts().max())})
    (C.RESULTS / "cohort_flow_d130.json").write_text(json.dumps(flow, indent=2))
    missing.to_csv(C.RESULTS / "d130_missingness.csv")
    H.mapping_table().to_csv(C.RESULTS / "comorbidity_mapping.csv", index=False)
    print(json.dumps(flow, indent=2))
    print(missing.head(10))
    print(out[H.COMORBIDITY_FEATURES].mean().round(4).sort_values(ascending=False))


if __name__ == "__main__":
    main()
