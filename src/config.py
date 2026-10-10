"""Central configuration: paths, seeds and study constants.

Every script imports from here so that a single edit changes the whole pipeline.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

D130_DIR = DATA_RAW / "diabetes130"
SYNTHEA_DIR = DATA_RAW / "synthea"
SYNTHEA_JAR = SYNTHEA_DIR / "synthea-with-dependencies.jar"
SYNTHEA_KEEP_MODULE = ROOT / "synthea" / "keep_diabetes.json"

SEED = 20260927  # global seed for splits, CV, bootstrap

# ----------------------------------------------------------------------------
# Diabetes 130-US Hospitals (UCI id 296)
# ----------------------------------------------------------------------------
D130_OFFICIAL_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
# GitHub mirrors used when the official UCI host is unreachable. Two independent
# uploads are listed; both were verified to be byte-identical (same SHA-256) and to
# match the published dataset invariants below.
D130_MIRRORS = [
    {
        "data": "https://raw.githubusercontent.com/rischanlab/"
        "Cleaning_diabetes_130_US_hospital_dataset/HEAD/diabetic_data.csv",
        "mapping": "https://raw.githubusercontent.com/rischanlab/"
        "Cleaning_diabetes_130_US_hospital_dataset/HEAD/IDs_mapping.csv",
    },
    {
        "data": "https://raw.githubusercontent.com/IT25101804/Group_58/HEAD/"
        "data/raw/diabetic_data.csv",
        "mapping": "https://raw.githubusercontent.com/IT25101804/Group_58/HEAD/"
        "data/raw/IDS_mapping.csv",
    },
]
D130_SHA256 = "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"
D130_MAPPING_SHA256 = "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e"
# Invariants published on the UCI page and in Strack et al. (2014).
D130_EXPECTED = {
    "n_rows": 101_766,
    "n_cols": 50,
    "n_patients": 71_518,
    "readmitted": {"NO": 54_864, ">30": 35_545, "<30": 11_357},
}
# discharge_disposition_id codes meaning death or hospice (IDs_mapping.csv):
# 11 Expired; 13 Hospice/home; 14 Hospice/medical facility;
# 19/20/21 Expired at home / in a medical facility / place unknown (Medicaid hospice).
D130_DEATH_HOSPICE_CODES = [11, 13, 14, 19, 20, 21]

# ----------------------------------------------------------------------------
# Synthea external cohort
# ----------------------------------------------------------------------------
SYNTHEA_RELEASE_URL = (
    "https://github.com/synthetichealth/synthea/releases/download/"
    "master-branch-latest/synthea-with-dependencies.jar"
)
# The "master-branch-latest" tag moves over time; this is the build used for the
# study (version.txt inside the jar reports git commit d9d07a6).
SYNTHEA_JAR_SHA256 = "018ad7f04f7aacb995804d7d4781c76d5fc714f7f23257ba50daa9eefae224ac"
SYNTHEA_JAR_VERSION = "d9d07a6"
# One run per state, two states per US census region, to obtain a multi-region cohort
# loosely mirroring the multi-hospital, national scope of diabetes-130. (The last four
# states were added by Amendment 1 (planned-care exclusion), before any external evaluation,
# because excluding planned-care stays roughly thirds the number of index stays.)
SYNTHEA_RUNS = [
    {"state": "Massachusetts", "seed": 1001},   # Northeast
    {"state": "Ohio", "seed": 1002},            # Midwest
    {"state": "Georgia", "seed": 1003},         # South
    {"state": "California", "seed": 1004},      # West
    {"state": "New York", "seed": 1005},        # Northeast
    {"state": "Illinois", "seed": 1006},        # Midwest
    {"state": "Texas", "seed": 1007},           # South
    {"state": "Washington", "seed": 1008},      # West
]
SYNTHEA_POP_PER_RUN = 2500  # living patients requested per run (deceased come on top)
SYNTHEA_AGE_RANGE = "35-100"  # age at the reference date; T2DM is rare below this
# Simulate up to 2009-01-01 so the exported window (1998-2008) matches the
# 1999-2008 period of diabetes-130 (and the COVID-19 module never activates).
SYNTHEA_END_DATE = "20090101"
SYNTHEA_YEARS_OF_HISTORY = 11  # 1998-2008: one extra year for the 365-day look-back
SYNTHEA_INDEX_START = "1999-01-01"  # first admission date eligible as an index stay
SYNTHEA_FOLLOWUP_END = "2009-01-01"  # end of simulated follow-up
SYNTHEA_CSV_FILES = "patients.csv,encounters.csv,conditions.csv,medications.csv,payers.csv"

# ----------------------------------------------------------------------------
# Modelling / evaluation
# ----------------------------------------------------------------------------
TEST_FRACTION_FOLDS = 5  # 1 of 5 grouped, stratified folds -> 20% test split
CV_FOLDS = 5
N_BOOTSTRAP = 1000
MIN_SUBGROUP_N = 100  # smallest subgroup (encounters) for which metrics are reported
MIN_SUBGROUP_EVENTS = 10  # ... and minimum number of readmissions in that subgroup
DCA_THRESHOLDS = [round(0.01 * i, 2) for i in range(1, 51)]  # 0.01 ... 0.50
