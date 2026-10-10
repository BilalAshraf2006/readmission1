"""Step 2 - generate the external synthetic cohort with Synthea.

One Synthea run per state (config.SYNTHEA_RUNS). Each run
  * uses a "keep module" (synthea/keep_diabetes.json, shipped with Synthea) so that only
    patients with an active type-2 diabetes diagnosis (SNOMED 44054006) are exported,
    i.e. the synthetic population is restricted to diabetic patients, like diabetes-130;
  * restricts age at the reference date to 35-100 (without an age floor, Synthea spends
    up to 1,000 attempts per child "slot" that can never develop type-2 diabetes);
  * ends the simulation on 2009-01-01 and exports 11 years of history, so index stays
    fall in 1999-2008 (the diabetes-130 period) with a full one-year look-back;
  * writes only the CSV tables needed downstream.

The run takes a few hours on 4 cores. Re-running with the same jar build and seeds
reproduces the cohort. Usage:  python src/generate_synthea.py [--force]
"""
import argparse
import json
import subprocess
import time

import config as C


def run_one(state, seed, force=False):
    out_dir = C.SYNTHEA_DIR / f"run_{state.lower().replace(' ', '_')}"
    done_flag = out_dir / "csv" / "patients.csv"
    if done_flag.exists() and not force:
        print(f"[synthea] {state}: output exists, skipping (use --force to regenerate)")
        return {"state": state, "seed": seed, "skipped": True, "out_dir": str(out_dir)}
    cmd = [
        "java", "-Xmx8g", "-jar", str(C.SYNTHEA_JAR),
        "-s", str(seed), "-cs", str(seed),
        "-p", str(C.SYNTHEA_POP_PER_RUN),
        "-a", C.SYNTHEA_AGE_RANGE,
        "-r", C.SYNTHEA_END_DATE, "-e", C.SYNTHEA_END_DATE,
        "-k", str(C.SYNTHEA_KEEP_MODULE),
        f"--exporter.baseDirectory={out_dir}",
        "--exporter.csv.export=true",
        "--exporter.fhir.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        f"--exporter.years_of_history={C.SYNTHEA_YEARS_OF_HISTORY}",
        f"--exporter.csv.included_files={C.SYNTHEA_CSV_FILES}",
        state,
    ]
    print("[synthea] running:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "synthea_stdout.log").write_text(proc.stdout + "\n" + proc.stderr)
    records = [ln for ln in proc.stdout.splitlines() if ln.startswith("Records:")]
    if proc.returncode != 0 or not done_flag.exists():
        raise RuntimeError(f"Synthea failed for {state}; see {out_dir}/synthea_stdout.log")
    print(f"[synthea] {state}: {records} in {elapsed / 60:.1f} min", flush=True)
    return {"state": state, "seed": seed, "cmd": cmd, "minutes": round(elapsed / 60, 1),
            "records_line": records[0] if records else None, "out_dir": str(out_dir)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    path = C.SYNTHEA_DIR / "generation_manifest.json"
    previous = json.loads(path.read_text()) if path.exists() else {"runs": []}
    by_state = {r["state"]: r for r in previous.get("runs", [])}
    manifest = {"jar_sha256_expected": C.SYNTHEA_JAR_SHA256,
                "jar_version": C.SYNTHEA_JAR_VERSION, "runs": []}
    for run in C.SYNTHEA_RUNS:
        rec = run_one(run["state"], run["seed"], args.force)
        if rec.get("skipped") and run["state"] in by_state:
            rec = by_state[run["state"]]  # keep the record of the run that made the data
        manifest["runs"].append(rec)
        path.write_text(json.dumps(manifest, indent=2))
    print("[synthea] all runs complete")
