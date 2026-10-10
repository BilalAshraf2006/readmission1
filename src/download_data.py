"""Step 1 - download the raw data.



Usage:  python src/download_data.py
"""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile

import pandas as pd

import config as C


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "readmission-study/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def verify_d130(csv_path):
    """Check the published invariants of diabetic_data.csv; raise if any differ."""
    df = pd.read_csv(csv_path, low_memory=False)
    exp = C.D130_EXPECTED
    checks = {
        "n_rows": (len(df), exp["n_rows"]),
        "n_cols": (df.shape[1], exp["n_cols"]),
        "n_patients": (df["patient_nbr"].nunique(), exp["n_patients"]),
        "readmitted": (df["readmitted"].value_counts().to_dict(), exp["readmitted"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f"diabetic_data.csv failed verification: {bad}")
    return {k: v[0] for k, v in checks.items()}


def download_d130():
    C.D130_DIR.mkdir(parents=True, exist_ok=True)
    data_path = C.D130_DIR / "diabetic_data.csv"
    map_path = C.D130_DIR / "IDs_mapping.csv"
    log = {"attempts": []}
    if not data_path.exists():
        try:  # 1) official UCI archive (zip containing both csv files)
            blob = fetch(C.D130_OFFICIAL_URL)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                names = z.namelist()
                inner = next(n for n in names if n.endswith("diabetic_data.csv"))
                data_path.write_bytes(z.read(inner))
                inner_map = next(n for n in names if n.lower().endswith("ids_mapping.csv"))
                map_path.write_bytes(z.read(inner_map))
            log["source"] = C.D130_OFFICIAL_URL
        except Exception as exc:  # noqa: BLE001 - report any failure and fall back
            log["attempts"].append({"url": C.D130_OFFICIAL_URL, "error": repr(exc)})
            print(f"[download] official UCI source failed: {exc!r}; trying mirrors")
            for mirror in C.D130_MIRRORS:
                try:
                    data_path.write_bytes(fetch(mirror["data"]))
                    map_path.write_bytes(fetch(mirror["mapping"]))
                    log["source"] = mirror["data"]
                    break
                except Exception as exc2:  # noqa: BLE001
                    log["attempts"].append({"url": mirror["data"], "error": repr(exc2)})
                    print(f"[download] mirror failed: {exc2!r}")
            else:
                sys.exit("Could not download diabetes-130 from UCI or any mirror.")
    log["sha256"] = sha256(data_path)
    log["sha256_matches_verified_copy"] = log["sha256"] == C.D130_SHA256
    log["mapping_sha256"] = sha256(map_path)
    log["invariants"] = verify_d130(data_path)
    if not log["sha256_matches_verified_copy"]:
        # The invariants passed, so the content is the dataset; a different hash usually
        # means different line endings or quoting in the source zip.
        print("[download] note: SHA-256 differs from the verified mirror copy, "
              "but all published invariants match.")
    print(f"[download] diabetes-130 OK: {log['invariants']}")
    return log


def download_synthea_jar():
    C.SYNTHEA_DIR.mkdir(parents=True, exist_ok=True)
    if not C.SYNTHEA_JAR.exists():
        print("[download] fetching Synthea jar (~200 MB) from GitHub releases ...")
        C.SYNTHEA_JAR.write_bytes(fetch(C.SYNTHEA_RELEASE_URL, timeout=600))
    digest = sha256(C.SYNTHEA_JAR)
    same = digest == C.SYNTHEA_JAR_SHA256
    if not same:
        print("[download] WARNING: Synthea jar differs from the build used in the study "
              f"({C.SYNTHEA_JAR_VERSION}); 'master-branch-latest' has moved. Results "
              "will not be bit-identical.")
    return {"url": C.SYNTHEA_RELEASE_URL, "sha256": digest, "matches_study_build": same}


if __name__ == "__main__":
    manifest = {"diabetes130": download_d130(), "synthea_jar": download_synthea_jar()}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
