"""Run the whole pipeline end to end.

    python run_all.py             # every step (Synthea generation is skipped if its output exists)
    python run_all.py --from train  # resume from a given step
    python run_all.py --use-committed-synthea  # skip Synthea; use data/external/synthea_cohort.csv.gz

Steps: download -> synthea -> cohort_d130 -> cohort_synthea -> train -> evaluate -> shift
       -> figures
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = [
    ("download", ["download_data.py"]),
    ("synthea", ["generate_synthea.py"]),
    ("cohort_d130", ["build_diabetes130.py"]),
    ("cohort_synthea", ["build_synthea.py"]),
    ("train", ["train.py"]),
    ("evaluate", ["evaluate.py"]),
    ("shift", ["shift.py"]),
    ("figures", ["figures.py"]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="download", choices=[s for s, _ in STEPS])
    ap.add_argument("--use-committed-synthea", action="store_true",
                    help="skip Synthea generation and use the committed harmonized cohort")
    args = ap.parse_args()
    names = [s for s, _ in STEPS]
    for name, cmd in STEPS[names.index(args.start):]:
        if args.use_committed_synthea and name == "synthea":
            continue
        if args.use_committed_synthea and name == "cohort_synthea":
            cmd = cmd + ["--from-committed"]
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, *cmd], cwd=SRC, check=True)
        print(f"=== {name} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
