"""Shared (harmonized) feature definitions for diabetes-130 and the Synthea cohort.

Only features that can be defined with the same meaning in both sources are used:
age, sex, length of stay, prior-year inpatient / emergency / outpatient visits,
number of diagnoses, number of medications and comorbidity flags.

Comorbidity flags are defined twice: from ICD-9-CM codes (diabetes-130 diag_1..diag_3)
and from SNOMED-CT codes (Synthea conditions active during the stay). Diabetes-130
stores ICD-9 codes at the 3-digit category level (only 250.xx keeps its decimals) and
drops leading zeros ("38" = 038), so every ICD-9 rule below is expressed on 3-digit
categories or on 250.x sub-codes. Categories whose 3-digit level mixes unrelated
conditions (e.g. 305 = drug abuse *and* tobacco use disorder, 995 = sepsis *and*
allergy, V42/V45 = any transplant/post-procedural state) are deliberately left out.
"""
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature lists (column names in the processed cohort files)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "age_years",          # midpoint of the 10-year age band
    "los_days",           # length of stay (days, 1-14)
    "n_inpatient_prior",  # inpatient stays in the 365 days before admission
    "n_emergency_prior",  # emergency visits in the 365 days before admission
    "n_outpatient_prior",  # outpatient visits in the 365 days before admission
    "n_diagnoses",        # number of diagnoses recorded for the stay (capped at 9)
    "n_medications",      # number of distinct medications during the stay
]
COUNT_FEATURES = ["n_inpatient_prior", "n_emergency_prior", "n_outpatient_prior"]

# name -> (label, ICD-9 rules, SNOMED codes, SNOMED description regex or None)
# ICD-9 rules: ("range", lo, hi) on the 3-digit numeric category (inclusive),
#              ("prefix", "250.4") on the normalized code string.
COMORBIDITIES = {
    "cm_heart_failure": (
        "Heart failure", [("range", 428, 428)],
        {"88805009", "84114007", "42343007", "85232009", "703272007"}, None),
    "cm_ischemic_heart": (
        "Ischemic heart disease / MI", [("range", 410, 414)],
        {"414545008", "22298006", "401303003", "401314000", "4557003", "53741008",
         "399211009", "399261000", "194828000"}, None),
    "cm_hypertension": (
        "Hypertension", [("range", 401, 405)],
        {"59621000", "38341003", "1201005"}, None),
    "cm_dysrhythmia": (
        "Cardiac dysrhythmia (incl. atrial fibrillation)", [("range", 427, 427)],
        {"49436004", "5370000", "698247007"}, None),
    "cm_valvular": (
        "Valvular heart disease", [("range", 394, 397), ("range", 424, 424)],
        {"60573004", "60234000", "48724000", "79619009", "111287006", "1231000119100"},
        None),
    "cm_cerebrovascular": (
        "Cerebrovascular disease", [("range", 430, 438)],
        {"230690007", "274100004", "422504002"}, None),
    "cm_chronic_lung": (
        "Chronic lung disease (COPD/asthma)", [("range", 490, 496)],
        {"185086009", "87433001", "195967001", "13645005", "233678006"}, None),
    "cm_renal": (
        "Chronic kidney disease", [("range", 403, 404), ("range", 581, 583),
                                   ("range", 585, 586), ("range", 588, 588),
                                   ("prefix", "250.4"), ("prefix", "V56")],
        {"431855005", "431856006", "433144002", "431857002", "46177005", "127013003",
         "90781000119102", "157141000119108", "698306007", "161665007", "213150003",
         "709044004"}, None),
    "cm_dm_complications": (
        "Diabetes with complications", [("prefix", f"250.{d}") for d in range(1, 10)],
        {"127013003", "90781000119102", "157141000119108", "368581000119106",
         "97331000119101", "1551000119108", "1501000119109", "60951000119105"},
        r"due to (type 2 )?diabetes|diabetic"),
    "cm_cancer": (
        "Malignancy", [("range", 140, 208), ("range", 230, 234)],
        {"126906006", "92691004"},
        r"malignant|carcinoma|cancer|leukemia|myeloma|lymphoma|sarcoma|melanoma"),
    "cm_obesity": (
        "Obesity", [("range", 278, 278)],
        {"162864005", "408512008"}, None),
    "cm_anemia": (
        "Anemia", [("range", 280, 285)],
        {"271737000"}, None),
    "cm_mood_disorder": (
        "Depression / mood disorder", [("range", 296, 296), ("range", 311, 311)],
        {"370143000", "36923009"}, None),
    "cm_dementia": (
        "Dementia / Alzheimer's", [("range", 290, 290), ("range", 294, 294),
                                   ("range", 331, 331)],
        {"26929004", "230265002"}, None),
    "cm_substance_use": (
        "Alcohol / drug dependence", [("range", 291, 292), ("range", 303, 304)],
        {"6525002", "7200002"}, None),
    "cm_hyperlipidemia": (
        "Hyperlipidemia", [("range", 272, 272)],
        {"55822004", "302870006"}, None),
    "cm_osteoarthritis": (
        "Osteoarthritis", [("range", 715, 715)],
        {"239873007", "239872002", "201834006", "396275006"}, None),
    "cm_sepsis_pneumonia": (
        "Septicemia / pneumonia", [("range", 38, 38), ("range", 480, 486)],
        {"91302008", "448813005", "76571007", "233604007", "10625071000119104"}, None),
}
COMORBIDITY_FEATURES = list(COMORBIDITIES)
BINARY_FEATURES = ["female"] + COMORBIDITY_FEATURES
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES
# Sensitivity analysis: only the features whose definitions are closest across the
# two sources (drops comorbidity flags, diagnosis/medication counts and outpatient
# visits, whose recording scope differs between an EHR extract and a simulated record).
CORE_FEATURES = ["age_years", "los_days", "n_inpatient_prior", "n_emergency_prior",
                 "female"]

# Subgroup (fairness) attributes - used for auditing only, never as model inputs
# (except sex and age, which are shared model features).
FAIRNESS_ATTRIBUTES = ["race", "sex", "age_band", "insurance"]
AGE_BAND_ORDER = ["<50", "50-59", "60-69", "70-79", "80+"]
RACE_ORDER = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"]
INSURANCE_ORDER = ["Medicare", "Medicaid", "Private/Other", "Self-pay", "Unknown"]
SEX_ORDER = ["Female", "Male"]


# ---------------------------------------------------------------------------
# ICD-9 side (diabetes-130)
# ---------------------------------------------------------------------------
def normalize_icd9(code):
    """Return (normalized string, 3-digit numeric category or None)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None, None
    s = str(code).strip()
    if not s:
        return None, None
    if s[0] in "VE":
        return s, None
    head = s.split(".")[0]
    try:
        cat = int(head)
    except ValueError:
        return s, None
    # diabetes-130 drops leading zeros (e.g. "38" is ICD-9 038 septicemia)
    s = f"{cat:03d}" + (s[len(head):] if "." in s else "")
    return s, cat


def icd9_matches(code, rules):
    s, cat = normalize_icd9(code)
    if s is None:
        return False
    for rule in rules:
        if rule[0] == "range" and cat is not None and rule[1] <= cat <= rule[2]:
            return True
        if rule[0] == "prefix" and s.startswith(rule[1]):
            return True
    return False


def icd9_flags(df, diag_cols=("diag_1", "diag_2", "diag_3")):
    """Comorbidity flags from the (up to) three ICD-9 diagnosis codes of an encounter."""
    out = pd.DataFrame(index=df.index)
    uniq = pd.unique(pd.concat([df[c] for c in diag_cols]).dropna().astype(str))
    for name, (_, rules, _, _) in COMORBIDITIES.items():
        hit = {u for u in uniq if icd9_matches(u, rules)}
        flag = np.zeros(len(df), dtype=bool)
        for c in diag_cols:
            flag |= df[c].astype(str).isin(hit).to_numpy()
        out[name] = flag.astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# SNOMED side (Synthea)
# ---------------------------------------------------------------------------
def snomed_flag_table(conditions_vocab):
    """Map every distinct Synthea condition (CODE, DESCRIPTION) to comorbidity flags.

    Returns a long table (code, description, flag) used both to build features and as
    an audit trail of exactly which SNOMED concepts fed each flag.
    """
    rows = []
    for code, desc in conditions_vocab[["CODE", "DESCRIPTION"]].drop_duplicates().itertuples(
            index=False):
        code_s = str(code)
        low = str(desc).lower()
        for name, (_, _, codes, pattern) in COMORBIDITIES.items():
            hit = code_s in codes
            if not hit and pattern is not None:
                # regex fallbacks apply only to diagnoses, never to 'suspected ...'
                # situations or screening findings
                is_disorder = low.endswith("(disorder)")
                hit = is_disorder and re.search(pattern, low) is not None \
                    and "suspected" not in low
            if hit:
                rows.append({"CODE": code_s, "DESCRIPTION": desc, "flag": name})
    return pd.DataFrame(rows, columns=["CODE", "DESCRIPTION", "flag"])


# ---------------------------------------------------------------------------
# Demographic / subgroup harmonization
# ---------------------------------------------------------------------------
def age_band_from_years(age):
    age = np.asarray(age, dtype=float)
    return pd.Series(np.select(
        [age < 50, age < 60, age < 70, age < 80], ["<50", "50-59", "60-69", "70-79"],
        default="80+"))


def d130_insurance(payer_code):
    """diabetes-130 payer_code -> insurance group. Only MC/MD/SP are unambiguous;
    all other codes (HM, BC, UN, CP, CM, OG, PO, DM, CH, WC, OT, MP, SI, FR) are
    commercial/managed-care/other plans and are pooled as Private/Other."""
    m = {"MC": "Medicare", "MD": "Medicaid", "SP": "Self-pay"}
    return payer_code.map(lambda x: "Unknown" if pd.isna(x) else m.get(x, "Private/Other"))


def synthea_insurance(payer_name):
    """Synthea payer name -> insurance group. Dual-eligible patients are grouped with
    Medicare because Medicare is the primary payer for dual-eligible beneficiaries, which
    is what a hospital's primary payer field records."""
    m = {"Medicare": "Medicare", "Dual Eligible": "Medicare", "Medicaid": "Medicaid",
         "NO_INSURANCE": "Self-pay"}
    return payer_name.map(lambda x: "Unknown" if pd.isna(x) else m.get(x, "Private/Other"))


def synthea_race(race, ethnicity):
    """Synthea race/ethnicity -> diabetes-130 race categories (which list Hispanic as a
    race category)."""
    base = race.map({"white": "Caucasian", "black": "AfricanAmerican", "asian": "Asian"})
    base = base.fillna("Other")
    return base.where(ethnicity != "hispanic", "Hispanic")


def mapping_table():
    """Human-readable comorbidity mapping for the README."""
    rows = []
    for name, (label, rules, codes, pattern) in COMORBIDITIES.items():
        icd = []
        for r in rules:
            if r[0] == "range":
                icd.append(f"{r[1]:03d}" if r[1] == r[2] else f"{r[1]:03d}-{r[2]:03d}")
            else:
                icd.append(r[1] + ".x" if not r[1].startswith("V") else r[1])
        rows.append({"flag": name, "label": label, "icd9_diabetes130": ", ".join(icd),
                     "snomed_codes_synthea": ", ".join(sorted(codes)),
                     "snomed_description_regex": pattern or ""})
    return pd.DataFrame(rows)
