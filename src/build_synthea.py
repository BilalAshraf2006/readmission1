"""Step 3b - derive the external (synthetic) cohort from the Synthea CSV export.

Index stays mirror the diabetes-130 inclusion criteria as closely as the data allow:
  * inpatient stays (overlapping/abutting inpatient encounters of the same patient,
    e.g. a ward stay and a nested ICU-transfer encounter, are merged into one stay so
    that a transfer is never counted as a readmission);
  * admitted 1999-01-01 or later (so the 365-day look-back lies inside the export) and
    discharged at least 30 days before the end of the simulation (complete follow-up);
  * "diabetic encounter": type-2 diabetes (SNOMED 44054006) diagnosed on or before the
    discharge date;
  * length of stay 1-14 days and at least one medication during the stay (diabetes-130
    required medications to be administered);
  * excluded: death during the stay (death date <= discharge date + 1 day) or discharge
    to hospice (a hospice encounter starting before discharge + 1 day) - the same rule
    as the discharge-disposition exclusion in diabetes-130;
  * excluded (Amendment 1): planned-care admissions - an injectable antineoplastic drug
    started during the stay (chemotherapy), or a drug-rehabilitation/detoxification or
    transplant admission (CMS "always planned" categories).

Label: 1 if the patient's next inpatient stay (any cause, any length) starts within
30 days of the index discharge, else 0. y_unplanned (sensitivity analysis f) also
requires that next stay not to be planned care.

Outputs: synthea.parquet (analysis cohort) and synthea_incl_planned.parquet (the
un-amended cohort with planned-care stays, sensitivity analysis e).

Usage:  python src/build_synthea.py [--runs-glob 'run_*'] [--out synthea.parquet]
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

import config as C
import harmonize as H

OUTPATIENT_CLASSES = {"ambulatory", "outpatient", "wellness", "urgentcare", "virtual"}
MERGE_GAP = pd.Timedelta(hours=1)   # stays abutting within 1 h are one stay
ED_TO_ADMIT_WINDOW = pd.Timedelta(hours=24)
DENTAL = re.compile(r"gingiv|dental|teeth|tooth|molar|caries|alveol|torus|tongue tie|"
                    r"jaw", re.I)
T2DM = 44054006
PLANNED_ENCOUNTERS = {"Drug rehabilitation and detoxification (regime/therapy)",
                      "Admission to surgical transplant department (procedure)"}
CHEMO = re.compile(r"cisplatin|carboplatin|paclitaxel|docetaxel|etoposide|doxorubicin|"
                   r"epirubicin|cyclophosphamide|oxaliplatin|leucovorin|bevacizumab|"
                   r"trastuzumab|gemcitabine|pemetrexed|vinorelbine|fluorouracil|"
                   r"irinotecan|topotecan|cytarabine|vincristine|ifosfamide|bleomycin|"
                   r"dacarbazine|daunorubicin|idarubicin|mitomycin|rituximab", re.I)
COMMITTED = C.ROOT / "data" / "external" / "synthea_cohort.csv.gz"


def ts(s):
    """Parse Synthea date/datetime strings to tz-naive UTC timestamps."""
    return pd.to_datetime(s, utc=True, format="ISO8601").dt.tz_localize(None)


def load_run(run_dir):
    csv = run_dir / "csv"
    tag = run_dir.name.replace("run_", "")
    pat = pd.read_csv(csv / "patients.csv", usecols=[
        "Id", "BIRTHDATE", "DEATHDATE", "RACE", "ETHNICITY", "GENDER", "STATE"])
    enc = pd.read_csv(csv / "encounters.csv", engine="pyarrow", usecols=[
        "Id", "START", "STOP", "PATIENT", "PAYER", "ENCOUNTERCLASS", "DESCRIPTION",
        "REASONDESCRIPTION"])
    cond = pd.read_csv(csv / "conditions.csv", engine="pyarrow", usecols=[
        "START", "STOP", "PATIENT", "CODE", "DESCRIPTION"])
    med = pd.read_csv(csv / "medications.csv", engine="pyarrow", usecols=[
        "START", "STOP", "PATIENT", "CODE", "DESCRIPTION"])
    pay = pd.read_csv(csv / "payers.csv", usecols=["Id", "NAME"])
    for df, cols in [(pat, ["BIRTHDATE", "DEATHDATE"]), (enc, ["START", "STOP"]),
                     (cond, ["START", "STOP"]), (med, ["START", "STOP"])]:
        for c in cols:
            df[c] = ts(df[c])
    # patient ids are UUIDs; prefix with the run so ids are unique across runs
    pat["Id"] = tag + ":" + pat["Id"]
    for df in (enc, cond, med):
        df["PATIENT"] = tag + ":" + df["PATIENT"]
    enc["PAYER_NAME"] = enc.PAYER.map(dict(zip(pay.Id, pay.NAME)))
    cond["CODE"] = cond.CODE.astype(str)
    return pat, enc, cond, med


def build_stays(enc):
    """Merge overlapping / abutting inpatient encounters into stays."""
    inp = enc[enc.ENCOUNTERCLASS == "inpatient"].sort_values(["PATIENT", "START"]).copy()
    prev_max_stop = inp.groupby("PATIENT").STOP.transform(lambda s: s.cummax().shift())
    new_stay = prev_max_stop.isna() | (inp.START > prev_max_stop + MERGE_GAP)
    inp["stay_no"] = new_stay.astype(int).groupby(inp.PATIENT).cumsum()
    inp["planned_enc"] = inp.DESCRIPTION.isin(PLANNED_ENCOUNTERS)
    gap_hours = ((inp.START - prev_max_stop).dt.total_seconds() / 3600)[prev_max_stop.notna()]
    stays = inp.groupby(["PATIENT", "stay_no"], sort=False).agg(
        admit=("START", "min"), discharge=("STOP", "max"), n_encounters=("Id", "size"),
        payer=("PAYER_NAME", "first"), reason=("REASONDESCRIPTION", "first"),
        planned_enc=("planned_enc", "max")).reset_index()
    stays = stays.sort_values(["PATIENT", "admit"]).reset_index(drop=True)
    stays["stay_id"] = stays.PATIENT + "#" + stays.stay_no.astype(str)
    return stays, gap_hours


def count_prior(stays, events, start_col="START"):
    """Number of events per stay with START in [admit - 365 d, admit)."""
    m = stays[["stay_id", "PATIENT", "admit"]].merge(
        events[["PATIENT", start_col]], on="PATIENT", how="inner")
    ok = (m[start_col] < m.admit) & (m[start_col] >= m.admit - pd.Timedelta(days=365))
    return m[ok].groupby("stay_id").size().reindex(stays.stay_id, fill_value=0).to_numpy()


def from_committed():
    """Rebuild data/processed/synthea.parquet from the committed harmonized cohort."""
    df = pd.read_csv(COMMITTED, parse_dates=["admit", "discharge"])
    for c in ["y", "y_unplanned"] + H.BINARY_FEATURES:
        df[c] = df[c].astype(np.int8)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.DATA_PROCESSED / "synthea_incl_planned.parquet", index=False)
    df[~df.planned].reset_index(drop=True).to_parquet(C.DATA_PROCESSED / "synthea.parquet",
                                                      index=False)
    print(f"[synthea] rebuilt cohorts from {COMMITTED} ({len(df):,} stays, "
          f"{int((~df.planned).sum()):,} unplanned)")


def main(runs_glob, out_name):
    runs = sorted(p for p in C.SYNTHEA_DIR.glob(runs_glob) if (p / "csv").exists())
    if not runs:
        raise SystemExit(f"no Synthea runs matching {runs_glob} under {C.SYNTHEA_DIR}")
    parts = [load_run(r) for r in runs]
    pat = pd.concat([p[0] for p in parts], ignore_index=True)
    enc = pd.concat([p[1] for p in parts], ignore_index=True)
    cond = pd.concat([p[2] for p in parts], ignore_index=True)
    med = pd.concat([p[3] for p in parts], ignore_index=True)
    print(f"[synthea] runs={[r.name for r in runs]} patients={len(pat):,} "
          f"encounters={len(enc):,} conditions={len(cond):,} medications={len(med):,}")

    stays, gap_hours = build_stays(enc)
    n_inp_enc = int((enc.ENCOUNTERCLASS == "inpatient").sum())
    stays = stays.merge(pat.rename(columns={"Id": "PATIENT"}), on="PATIENT", how="left")

    # label: next stay (any) starting within 30 days of discharge
    nxt = stays.groupby("PATIENT").admit.shift(-1)
    days_to_next = (nxt.dt.normalize() - stays.discharge.dt.normalize()).dt.days
    stays["days_to_next_admit"] = days_to_next
    stays["y"] = ((days_to_next >= 0) & (days_to_next <= 30)).astype(np.int8)

    # planned care (Amendment 1): chemotherapy given during the stay, or a rehab/detox or
    # transplant admission. Needed for every stay because y_unplanned looks at the next one.
    chemo = med[med.DESCRIPTION.str.contains(CHEMO, na=False)
                & med.DESCRIPTION.str.contains("inject", case=False, na=False)]
    cm = stays[["stay_id", "PATIENT", "admit", "discharge"]].merge(
        chemo[["PATIENT", "START"]], on="PATIENT")
    chemo_stays = set(cm.stay_id[(cm.START >= cm.admit) & (cm.START <= cm.discharge)])
    stays["planned_chemo"] = stays.stay_id.isin(chemo_stays)
    stays["planned"] = stays.planned_chemo | stays.planned_enc
    next_planned = stays.groupby("PATIENT").planned.shift(-1).fillna(False).astype(bool)
    stays["y_unplanned"] = (stays.y.astype(bool) & ~next_planned).astype(np.int8)

    # stay-level attributes
    stays["los_days"] = (stays.discharge.dt.normalize() - stays.admit.dt.normalize()
                         ).dt.days.clip(lower=1).astype(float)
    age = (stays.admit - stays.BIRTHDATE).dt.days / 365.25
    stays["age_exact"] = age
    stays["age_years"] = (np.minimum(np.floor(age / 10) * 10, 90) + 5).astype(float)
    dm_onset = cond[cond.CODE == str(T2DM)].groupby("PATIENT").START.min()
    stays["dm_onset"] = stays.PATIENT.map(dm_onset)
    hospice = enc[enc.ENCOUNTERCLASS == "hospice"][["PATIENT", "START"]]
    h = stays[["stay_id", "PATIENT", "admit", "discharge"]].merge(hospice, on="PATIENT")
    h = h[(h.START >= h.admit) & (h.START <= h.discharge + pd.Timedelta(days=1))]
    stays["to_hospice"] = stays.stay_id.isin(h.stay_id)
    stays["died_in_stay"] = stays.DEATHDATE.notna() & (
        stays.DEATHDATE <= stays.discharge.dt.normalize() + pd.Timedelta(days=1))
    stays["died_30d_after"] = stays.DEATHDATE.notna() & ~stays.died_in_stay & (
        stays.DEATHDATE <= stays.discharge.dt.normalize() + pd.Timedelta(days=30))

    # --- prior-year utilization -------------------------------------------
    stays["n_inpatient_prior"] = count_prior(stays, stays.rename(columns={"admit": "START"})
                                             [["PATIENT", "START"]]).astype(float)
    # The remaining features are only needed for candidate index stays (inside the
    # admission window with complete follow-up); computing them for every stay since
    # birth would multiply the merges below for no benefit.
    fu_end = pd.Timestamp(C.SYNTHEA_FOLLOWUP_END)
    win = (stays.admit >= pd.Timestamp(C.SYNTHEA_INDEX_START)) & (
        stays.discharge <= fu_end - pd.Timedelta(days=30))
    sw = stays.loc[win, ["stay_id", "PATIENT", "admit", "discharge"]]
    ed = enc[enc.ENCOUNTERCLASS == "emergency"][["PATIENT", "START", "STOP"]]
    # an ED visit that ends in an admission is part of that admission, not a separate
    # emergency visit (as in hospital encounter data)
    e2 = ed.reset_index().merge(stays[["PATIENT", "admit"]], on="PATIENT")
    to_admit = e2[(e2.admit >= e2.START) & (e2.admit - e2.START <= ED_TO_ADMIT_WINDOW)]
    ed = ed.drop(index=to_admit["index"].unique())
    stays.loc[win, "n_emergency_prior"] = count_prior(sw, ed).astype(float)
    op = enc[enc.ENCOUNTERCLASS.isin(OUTPATIENT_CLASSES)][["PATIENT", "START"]]
    stays.loc[win, "n_outpatient_prior"] = count_prior(sw, op).astype(float)

    # --- medications active during the stay --------------------------------
    m = sw.merge(med, on="PATIENT")
    m = m[(m.START <= m.discharge) & (m.STOP.isna() | (m.STOP >= m.admit))]
    stays.loc[win, "n_medications"] = m.groupby("stay_id").CODE.nunique().reindex(
        sw.stay_id, fill_value=0).to_numpy().astype(float)

    # --- conditions active during the stay: flags and diagnosis count -------
    vocab = cond[["CODE", "DESCRIPTION"]].drop_duplicates("CODE")
    flag_tab = H.snomed_flag_table(vocab)
    flagged_codes = set(flag_tab.CODE)
    vocab["countable"] = vocab.apply(
        lambda r: (str(r.DESCRIPTION).endswith("(disorder)")
                   and not DENTAL.search(str(r.DESCRIPTION))) or r.CODE in flagged_codes,
        axis=1)
    countable = set(vocab.CODE[vocab.countable])
    c = sw.merge(cond[["PATIENT", "START", "STOP", "CODE"]], on="PATIENT")
    c = c[(c.START <= c.discharge.dt.normalize())
          & (c.STOP.isna() | (c.STOP >= c.admit.dt.normalize()))]
    # capped at 9, the diabetes-130 recording limit
    stays.loc[win, "n_diagnoses"] = c[c.CODE.isin(countable)].groupby("stay_id").CODE \
        .nunique().reindex(sw.stay_id, fill_value=0).clip(upper=9).to_numpy().astype(float)
    for flag in H.COMORBIDITY_FEATURES:
        codes = set(flag_tab.CODE[flag_tab.flag == flag])
        hit = set(c.stay_id[c.CODE.isin(codes)])
        stays[flag] = stays.stay_id.isin(hit).astype(np.int8)

    stays["female"] = (stays.GENDER == "F").astype(np.int8)
    stays["sex"] = stays.GENDER.map({"F": "Female", "M": "Male"})
    stays["race"] = H.synthea_race(stays.RACE, stays.ETHNICITY)
    stays["insurance"] = H.synthea_insurance(stays.payer)
    stays["age_band"] = H.age_band_from_years(stays.age_exact.to_numpy()).to_numpy()

    # --- cohort flow ----------------------------------------------------------
    flow = [{"step": "Synthea patients (all with type-2 diabetes by end of simulation)",
             "patients": int(pat.Id.nunique())},
            {"step": "Inpatient encounters -> merged inpatient stays",
             "inpatient_encounters": n_inp_enc, "stays": len(stays),
             "patients": int(stays.PATIENT.nunique())}]
    keep = pd.Series(True, index=stays.index)
    steps = [
        ("Excluded: admitted before 1999-01-01 (incomplete 365-day look-back)",
         stays.admit < pd.Timestamp(C.SYNTHEA_INDEX_START)),
        ("Excluded: discharged within 30 days of the end of simulation "
         "(incomplete follow-up)", stays.discharge > fu_end - pd.Timedelta(days=30)),
        ("Excluded: type-2 diabetes not yet diagnosed at discharge (not a diabetic "
         "encounter)", ~(stays.dm_onset <= stays.discharge)),
        ("Excluded: length of stay > 14 days", stays.los_days > 14),
        ("Excluded: no medication during the stay", stays.n_medications < 1),
        ("Excluded: died during the stay or discharged to hospice",
         stays.died_in_stay | stays.to_hospice),
        ("Excluded: planned-care admission (chemotherapy, rehabilitation/detox, "
         "transplant; Amendment 1)", stays.planned),
    ]
    for label, mask in steps:
        drop = keep & mask
        rec = {"step": label, "excluded": int(drop.sum())}
        if "died" in label:
            rec["excluded_died"] = int((drop & stays.died_in_stay).sum())
            rec["excluded_hospice_only"] = int((drop & ~stays.died_in_stay).sum())
            rec["excluded_readmit_30d"] = int(stays.y[drop].sum())
        if "planned" in label:
            rec["excluded_chemotherapy"] = int((drop & stays.planned_chemo).sum())
            rec["excluded_rehab_or_transplant_only"] = int(
                (drop & ~stays.planned_chemo).sum())
            rec["excluded_readmit_30d"] = int(stays.y[drop].sum())
            keep_incl_planned = keep.copy()  # the un-amended cohort (sensitivity e)
        keep &= ~mask
        rec.update({"stays": int(keep.sum()), "patients": int(stays.PATIENT[keep].nunique())})
        flow.append(rec)
    co = stays[keep].copy()
    co_ip = stays[keep_incl_planned].copy()
    flow.append({"step": "Analysis cohort", "stays": len(co),
                 "patients": int(co.PATIENT.nunique()), "readmit_30d": int(co.y.sum()),
                 "prevalence": round(float(co.y.mean()), 4),
                 "readmit_30d_unplanned_outcome": int(co.y_unplanned.sum()),
                 "died_within_30d_after_discharge_without_readmission":
                     int((co.died_30d_after & (co.y == 0)).sum()),
                 "stays_per_patient_max": int(co.PATIENT.value_counts().max()),
                 "incl_planned_cohort": {"stays": len(co_ip),
                                         "patients": int(co_ip.PATIENT.nunique()),
                                         "readmit_30d": int(co_ip.y.sum()),
                                         "prevalence": round(float(co_ip.y.mean()), 4)}})

    def to_out(co):
        return pd.DataFrame({
            "encounter_id": co.stay_id, "patient_id": "syn_" + co.PATIENT, "y": co.y,
            "y_unplanned": co.y_unplanned, "planned": co.planned,
            **{f: co[f] for f in H.NUMERIC_FEATURES + H.BINARY_FEATURES},
            "race": co.race, "sex": co.sex, "age_band": co.age_band,
            "insurance": co.insurance, "state": co.STATE, "admit": co.admit,
            "discharge": co.discharge, "reason": co.reason,
            "n_encounters_merged": co.n_encounters, "died_30d_after": co.died_30d_after,
            "days_to_next_admit": co.days_to_next_admit}).reset_index(drop=True)

    out, out_ip = to_out(co), to_out(co_ip)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(C.DATA_PROCESSED / out_name, index=False)
    out_ip.to_parquet(C.DATA_PROCESSED / out_name.replace(".parquet", "_incl_planned.parquet"),
                      index=False)
    if out_name == "synthea.parquet":  # committed copy for re-analysis without Synthea
        COMMITTED.parent.mkdir(parents=True, exist_ok=True)
        out_ip.to_csv(COMMITTED, index=False, compression="gzip")  # superset, has 'planned'

    stem = out_name.replace(".parquet", "")
    (C.RESULTS / f"cohort_flow_{stem}.json").write_text(json.dumps(flow, indent=2))
    audit = flag_tab.merge(
        c[c.stay_id.isin(co.stay_id)].groupby("CODE").stay_id.nunique().rename("n_stays"),
        left_on="CODE", right_index=True, how="left").fillna({"n_stays": 0})
    audit.sort_values(["flag", "n_stays"], ascending=[True, False]).to_csv(
        C.RESULTS / f"{stem}_snomed_flag_audit.csv", index=False)
    gaps = gap_hours.describe(percentiles=[.05, .1, .25, .5]).round(2).to_dict()
    gaps["n_overlapping_or_abutting_(<=1h)"] = int((gap_hours <= 1).sum())
    (C.RESULTS / f"{stem}_inpatient_gap_hours.json").write_text(json.dumps(gaps, indent=2))
    print(json.dumps(flow, indent=2))
    print(out[H.FEATURES].describe().T[["mean", "50%", "max"]].round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="run_*")
    ap.add_argument("--out", default="synthea.parquet")
    ap.add_argument("--from-committed", action="store_true",
                    help="use data/external/synthea_cohort.csv.gz instead of raw Synthea CSVs")
    a = ap.parse_args()
    if a.from_committed:
        from_committed()
    else:
        main(a.runs_glob, a.out)
