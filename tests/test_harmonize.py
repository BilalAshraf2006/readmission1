import pandas as pd

import harmonize as H


def test_normalize_icd9_restores_leading_zeros():
    assert H.normalize_icd9("38") == ("038", 38)
    assert H.normalize_icd9("250.83") == ("250.83", 250)
    assert H.normalize_icd9("V56") == ("V56", None)
    assert H.normalize_icd9(None) == (None, None)


def test_icd9_flag_rules():
    rules = lambda name: H.COMORBIDITIES[name][1]  # noqa: E731
    assert H.icd9_matches("428", rules("cm_heart_failure"))
    assert H.icd9_matches("38", rules("cm_sepsis_pneumonia"))
    assert H.icd9_matches("250.41", rules("cm_renal"))
    assert H.icd9_matches("250.41", rules("cm_dm_complications"))
    assert not H.icd9_matches("250.02", rules("cm_dm_complications"))  # uncomplicated
    assert H.icd9_matches("V56", rules("cm_renal"))
    assert not H.icd9_matches("305", rules("cm_substance_use"))  # ambiguous: tobacco


def test_icd9_flags_dataframe():
    df = pd.DataFrame({"diag_1": ["428", "250.6"], "diag_2": ["401", None],
                       "diag_3": [None, "585"]})
    fl = H.icd9_flags(df)
    assert fl.loc[0, "cm_heart_failure"] == 1 and fl.loc[0, "cm_hypertension"] == 1
    assert fl.loc[1, "cm_dm_complications"] == 1 and fl.loc[1, "cm_renal"] == 1
    assert fl.loc[1, "cm_heart_failure"] == 0


def test_snomed_flags_codes_and_regex():
    vocab = pd.DataFrame({"CODE": ["88805009", "368581000119106", "162573006", "254637007"],
                          "DESCRIPTION": ["Chronic congestive heart failure (disorder)",
                                          "Neuropathy due to type 2 diabetes mellitus (disorder)",
                                          "Suspected lung cancer (situation)",
                                          "Non-small cell lung cancer (disorder)"]})
    tab = H.snomed_flag_table(vocab)
    flags = tab.groupby("CODE").flag.apply(set).to_dict()
    assert "cm_heart_failure" in flags["88805009"]
    assert "cm_dm_complications" in flags["368581000119106"]
    assert "162573006" not in flags  # 'suspected' is not a diagnosis
    assert "cm_cancer" in flags["254637007"]


def test_audit_attribute_mappings():
    ins = H.d130_insurance(pd.Series(["MC", "MD", "SP", "BC", None]))
    assert ins.tolist() == ["Medicare", "Medicaid", "Self-pay", "Private/Other", "Unknown"]
    syn = H.synthea_insurance(pd.Series(["Dual Eligible", "NO_INSURANCE", "Aetna"]))
    assert syn.tolist() == ["Medicare", "Self-pay", "Private/Other"]
    race = H.synthea_race(pd.Series(["white", "black", "white", "native"]),
                          pd.Series(["nonhispanic", "nonhispanic", "hispanic", "nonhispanic"]))
    assert race.tolist() == ["Caucasian", "AfricanAmerican", "Hispanic", "Other"]
