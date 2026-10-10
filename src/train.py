"""Step 4 - grouped train/test split, CV tuning, final fit, and test-set predictions.

Models (id = "<train source>|<feature set>|<algorithm>"):
  d130|primary|{lr,hgb}     main models (shared harmonized features); also scored on
                            the un-amended Synthea cohort (incl. planned care)
  d130|core|{lr,hgb}        sensitivity (a): definitionally closest features only
  d130|full|{lr,hgb}        sensitivity (d): all usable diabetes-130 columns (internal only)
  synthea|primary|{lr,hgb}  transfer matrix (trained on the Synthea 80% split)
  synthea|core|{lr,hgb}     transfer matrix, core features

Outputs: results/models.json, results/tuning_*.csv, results/predictions.parquet,
         results/splits.json, models/*.joblib

Fitted models are cached in models/ under a hash of (training data, feature spec, grid,
seed, scikit-learn version); a re-run with identical inputs reloads them instead of
re-tuning. `--only-d130` tunes and caches the diabetes-130-trained models without
needing the Synthea cohort.
Usage:  python src/train.py [--only-d130]
"""
import argparse
import hashlib
import json
import shutil
import time

import joblib
import pandas as pd
import sklearn

import config as C
import harmonize as H
import models as MD

AUDIT = ["race", "sex", "age_band", "insurance"]
FULL_EXTRA_NUM = ["x_num_lab_procedures", "x_num_procedures", "x_diag_missing_n"]


def feature_specs(d130_cols):
    cat_full = [c for c in d130_cols if c.startswith("x_") and c not in FULL_EXTRA_NUM]
    return {
        "primary": {"numeric": H.NUMERIC_FEATURES, "counts": H.COUNT_FEATURES,
                    "binary": H.BINARY_FEATURES},
        "core": {"numeric": [f for f in H.CORE_FEATURES if f != "female"],
                 "counts": ["n_inpatient_prior", "n_emergency_prior"], "binary": ["female"]},
        "full": {"numeric": H.NUMERIC_FEATURES + FULL_EXTRA_NUM, "counts": H.COUNT_FEATURES,
                 "binary": H.BINARY_FEATURES, "categorical": cat_full},
    }


def spec_columns(spec):
    return list(dict.fromkeys(spec["numeric"] + spec["counts"] + spec["binary"]
                              + spec.get("categorical", [])))


def fit_or_load(mid, kind, spec, train, cols):
    """Tune + fit, or reload an identical earlier fit from the cache in models/."""
    stem = mid.replace("|", "_")
    grid = MD.LR_GRID if kind == "lr" else MD.HGB_GRID
    h = hashlib.sha256(pd.util.hash_pandas_object(
        train[cols + ["y", "patient_id"]], index=False).to_numpy().tobytes())
    h.update(json.dumps({"spec": spec, "grid": grid, "seed": C.SEED, "folds": C.CV_FOLDS,
                         "sklearn": sklearn.__version__}, sort_keys=True).encode())
    key = h.hexdigest()
    model_path, meta_path = C.MODELS / f"{stem}.joblib", C.MODELS / f"{stem}.meta.json"
    tuning_cache = C.MODELS / f"tuning_{stem}.csv"
    if model_path.exists() and meta_path.exists() and tuning_cache.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("key") == key:
            shutil.copy(tuning_cache, C.RESULTS / f"tuning_{stem}.csv")
            print(f"[train] {mid}: cache hit, reusing fitted model", flush=True)
            return joblib.load(model_path), meta["info"]
    print(f"[train] {mid}: tuning on {len(train):,} rows ...", flush=True)
    t0 = time.time()
    best, table, oof = MD.tune(kind, spec, train[cols], train.y, train.patient_id,
                               C.CV_FOLDS, C.SEED)
    thr = MD.youden_threshold(train.y, oof)
    model = MD.make_model(kind, spec, best, C.SEED).fit(train[cols], train.y)
    best_row = table.loc[table.cv_logloss.idxmin()]
    info = {"features": cols, "best_params": best, "threshold": thr,
            "cv_logloss": float(best_row.cv_logloss), "cv_auroc": float(best_row.cv_auroc),
            "cv_auroc_sd": float(best_row.cv_auroc_sd), "n_train": len(train),
            "grid_size": len(table), "minutes": round((time.time() - t0) / 60, 1)}
    joblib.dump(model, model_path)
    table.to_csv(tuning_cache, index=False)
    shutil.copy(tuning_cache, C.RESULTS / f"tuning_{stem}.csv")
    meta_path.write_text(json.dumps({"key": key, "info": info}, indent=2))
    print(f"[train] {mid}: best={best} thr={thr:.4f} cv_auroc={info['cv_auroc']:.4f} "
          f"({info['minutes']} min)", flush=True)
    return model, info


def main(only_d130=False):
    t_start = time.time()
    C.RESULTS.mkdir(exist_ok=True); C.MODELS.mkdir(exist_ok=True)
    d130 = pd.read_parquet(C.DATA_PROCESSED / "d130.parquet")
    specs = feature_specs(d130.columns)
    d_tr, d_te = MD.grouped_split(d130.y, d130.patient_id, C.TEST_FRACTION_FOLDS, C.SEED)
    if only_d130:
        for fs in ["primary", "core", "full"]:
            for kind in ["lr", "hgb"]:
                fit_or_load(f"d130|{fs}|{kind}", kind, specs[fs],
                            d130.iloc[d_tr].reset_index(drop=True), spec_columns(specs[fs]))
        print(f"[train] diabetes-130 models cached in {(time.time() - t_start) / 60:.1f} min")
        return
    syn = pd.read_parquet(C.DATA_PROCESSED / "synthea.parquet")
    # un-amended external cohort incl. planned-care stays (sensitivity analysis e)
    syn_ip = pd.read_parquet(C.DATA_PROCESSED / "synthea_incl_planned.parquet")

    # ---- grouped, stratified 80/20 splits (no patient in both) -------------
    s_tr, s_te = MD.grouped_split(syn.y, syn.patient_id, C.TEST_FRACTION_FOLDS, C.SEED)
    for a, b, name in [(d_tr, d_te, "d130"), (s_tr, s_te, "synthea")]:
        df = d130 if name == "d130" else syn
        overlap = set(df.patient_id.iloc[a]) & set(df.patient_id.iloc[b])
        assert not overlap, f"patient leakage in {name} split"
    splits = {
        "d130": {"train_rows": len(d_tr), "test_rows": len(d_te),
                 "train_patients": int(d130.patient_id.iloc[d_tr].nunique()),
                 "test_patients": int(d130.patient_id.iloc[d_te].nunique()),
                 "train_prev": float(d130.y.iloc[d_tr].mean()),
                 "test_prev": float(d130.y.iloc[d_te].mean()), "patient_overlap": 0},
        "synthea": {"train_rows": len(s_tr), "test_rows": len(s_te),
                    "train_patients": int(syn.patient_id.iloc[s_tr].nunique()),
                    "test_patients": int(syn.patient_id.iloc[s_te].nunique()),
                    "train_prev": float(syn.y.iloc[s_tr].mean()),
                    "test_prev": float(syn.y.iloc[s_te].mean()), "patient_overlap": 0},
    }
    (C.RESULTS / "splits.json").write_text(json.dumps(splits, indent=2))
    print(json.dumps(splits, indent=2))

    d130_test = d130.iloc[d_te].reset_index(drop=True)
    syn_test = syn.iloc[s_te].reset_index(drop=True)
    test_sets = {"d130_test": d130_test, "synthea_test": syn_test, "synthea_all": syn,
                 "synthea_incl_planned": syn_ip}

    plan = [("d130", "primary", ["d130_test", "synthea_all", "synthea_test",
                                 "synthea_incl_planned"]),
            ("d130", "core", ["d130_test", "synthea_all", "synthea_test"]),
            ("synthea", "primary", ["synthea_test", "d130_test"]),
            ("synthea", "core", ["synthea_test", "d130_test"]),
            ("d130", "full", ["d130_test"])]
    registry, preds = {}, []
    for source, fs, targets in plan:
        train = (d130.iloc[d_tr] if source == "d130" else syn.iloc[s_tr]).reset_index(drop=True)
        spec = specs[fs]
        cols = spec_columns(spec)
        for kind in ["lr", "hgb"]:
            mid = f"{source}|{fs}|{kind}"
            model, info = fit_or_load(mid, kind, spec, train, cols)
            registry[mid] = {"train_source": source, "feature_set": fs, "algorithm": kind,
                             **info}
            for t in targets:
                df = test_sets[t]
                if fs == "full" and t != "d130_test":
                    continue
                p = model.predict_proba(df[cols])[:, 1]
                keep = ["encounter_id", "patient_id", "y"] + AUDIT + (
                    ["died_30d_after", "y_unplanned", "planned"] if "synthea" in t else [])
                out = df[keep].copy()
                out["p"] = p; out["model_id"] = mid; out["test_set"] = t
                preds.append(out)
            (C.RESULTS / "models.json").write_text(json.dumps(registry, indent=2))
    allp = pd.concat(preds, ignore_index=True)
    allp["died_30d_after"] = allp["died_30d_after"].fillna(False).astype(bool)
    allp["planned"] = allp["planned"].fillna(False).astype(bool)
    allp.to_parquet(C.RESULTS / "predictions.parquet", index=False)
    print(f"[train] done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-d130", action="store_true")
    main(ap.parse_args().only_d130)
