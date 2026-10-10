"""Portfolio artefact: one-page evidence of the model-training work.

Panels (all read from the pipeline's own outputs in results/):
  A  gradient boosting: CV log loss of all 24 settings tried, grouped by model size
  B  logistic regression: CV log loss for each C tried
  C  internal calibration of the final models (no class re-balancing was used)
  D  AUROC: cross-validation vs internal test vs external (Synthea)

Output: figures/evidence_training.png
Usage:  python src/training_evidence.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from figures import AXIS, BLUE, INK2, MUTED, ORANGE, SURFACE  # noqa: E402

LR_ID, GB_ID = "d130|primary|lr", "d130|primary|hgb"


def main():
    tun_gb = pd.read_csv(C.RESULTS / "tuning_d130_primary_hgb.csv")
    tun_lr = pd.read_csv(C.RESULTS / "tuning_d130_primary_lr.csv")
    reg = json.loads((C.RESULTS / "models.json").read_text())
    splits = json.loads((C.RESULTS / "splits.json").read_text())["d130"]
    met = pd.read_csv(C.RESULTS / "metrics_overall.csv")
    cal = pd.read_csv(C.RESULTS / "calibration_curves.csv")

    best_gb, best_lr = tun_gb.cv_logloss.min(), tun_lr.cv_logloss.min()
    lo = min(best_gb, best_lr)
    hi = max(tun_gb.cv_logloss.max(), tun_lr.cv_logloss.max())
    ylim = (lo - 0.15 * (hi - lo), hi + 0.25 * (hi - lo))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Evidence of model training: 30-day readmission project",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")

    # A: gradient boosting grid, grouped by model size
    a = ax[0, 0]
    sizes = [(200, 15), (200, 31), (400, 15), (400, 31)]
    for i, (it, leaves) in enumerate(sizes):
        g = tun_gb[(tun_gb.max_iter == it) & (tun_gb.max_leaf_nodes == leaves)]
        x = i + np.linspace(-0.2, 0.2, len(g))
        a.scatter(x, g.cv_logloss, color=ORANGE, s=22, zorder=3)
    b_row = tun_gb.loc[tun_gb.cv_logloss.idxmin()]
    bi = sizes.index((b_row.max_iter, b_row.max_leaf_nodes))
    g = tun_gb[(tun_gb.max_iter == b_row.max_iter)
               & (tun_gb.max_leaf_nodes == b_row.max_leaf_nodes)].reset_index(drop=True)
    bx = bi + np.linspace(-0.2, 0.2, len(g))[g.cv_logloss.idxmin()]
    a.scatter([bx], [best_gb], marker="*", s=220, color=ORANGE, edgecolor="black",
              zorder=4, label="chosen setting")
    a.axhline(best_lr, color=BLUE, ls="--", lw=1, label="best logistic regression")
    a.set_xticks(range(4))
    a.set_xticklabels([f"{it} trees\n{leaves} leaves" for it, leaves in sizes])
    a.set_ylim(ylim); a.set_ylabel("cross-validated log loss (lower = better)")
    a.legend(fontsize=8, frameon=False, loc="upper left")
    a.set_title(f"A. Gradient boosting: all {len(tun_gb)} settings tried",
                loc="left", fontsize=10)
    a.text(0.98, 0.04, f"chosen: {int(b_row.max_iter)} trees, "
                       f"{int(b_row.max_leaf_nodes)} leaves,\n"
                       f"min leaf {int(b_row.min_samples_leaf)}, "
                       f"L2 {b_row.l2_regularization:g}\n"
                       "bigger models scored worse",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8.5,
           bbox=dict(fc="white", ec=AXIS))

    # B: logistic regression, C
    b = ax[0, 1]
    b.plot(tun_lr.C, tun_lr.cv_logloss, "-o", color=BLUE, ms=5, zorder=3)
    c_best = tun_lr.loc[tun_lr.cv_logloss.idxmin(), "C"]
    b.scatter([c_best], [best_lr], marker="*", s=220, color=BLUE, edgecolor="black",
              zorder=4, label=f"chosen C = {c_best:g}")
    b.axhline(best_gb, color=ORANGE, ls="--", lw=1, label="best gradient boosting")
    b.set_xscale("log"); b.set_ylim(ylim)
    b.set_xlabel("C (inverse regularization strength)")
    b.set_ylabel("cross-validated log loss (lower = better)")
    b.legend(fontsize=8, frameon=False, loc="center right")
    b.set_title(f"B. Logistic regression: all {len(tun_lr)} settings tried",
                loc="left", fontsize=10)
    b.text(0.02, 0.96,
           "5-fold CV grouped by patient, chosen by log loss\n"
           f"training split: {splits['train_rows']:,} encounters, "
           f"{splits['train_patients']:,} patients\n"
           f"readmitted: {splits['train_prev']:.1%}; no re-sampling or class weights\n"
           f"threshold from training out-of-fold predictions:\n"
           f"LR {reg[LR_ID]['threshold']:.3f}, GB {reg[GB_ID]['threshold']:.3f} "
           "(fixed before testing)",
           transform=b.transAxes, ha="left", va="top", fontsize=8.5,
           bbox=dict(fc="white", ec=AXIS))

    # C: internal calibration
    c = ax[1, 0]
    top = 0
    for mid, col, name in [(LR_ID, BLUE, "Logistic regression"),
                           (GB_ID, ORANGE, "Gradient boosting")]:
        cc = cal[(cal.model_id == mid) & (cal.test_set == "d130_test")]
        r = met[(met.model_id == mid) & (met.test_set == "d130_test")].iloc[0]
        c.errorbar(cc.mean_pred, cc.observed,
                   yerr=[cc.observed - cc.obs_lo, cc.obs_hi - cc.observed],
                   fmt="o-", color=col, ms=4, lw=1.2, capsize=2,
                   label=f"{name}: slope {r.cal_slope:.2f}, O/E {r.oe_ratio:.2f}")
        top = max(top, cc.obs_hi.max(), cc.mean_pred.max())
    top = np.ceil(top * 20) / 20
    c.plot([0, top], [0, top], color=INK2, lw=1, ls=":", label="perfect calibration")
    c.set_xlim(0, top); c.set_ylim(0, top)
    c.set_xlabel("predicted risk (decile mean)")
    c.set_ylabel("observed readmission rate")
    c.legend(fontsize=8, frameon=False, loc="upper left")
    c.set_title("C. Internal test: predicted risks match observed rates",
                loc="left", fontsize=10)

    # D: CV vs internal test vs external
    d = ax[1, 1]
    cats = ["Cross-validation\n(training, out-of-fold)", "Internal test\n(diabetes-130)",
            "External\n(Synthea)"]
    for k, (mid, col, name) in enumerate([(LR_ID, BLUE, "Logistic regression"),
                                          (GB_ID, ORANGE, "Gradient boosting")]):
        off = -0.12 if k == 0 else 0.12
        tun = tun_lr if mid == LR_ID else tun_gb
        best = tun.loc[tun.cv_logloss.idxmin()]
        vals = [(best.cv_auroc, best.cv_auroc - best.cv_auroc_sd,
                 best.cv_auroc + best.cv_auroc_sd)]
        for ts in ["d130_test", "synthea_all"]:
            r = met[(met.model_id == mid) & (met.test_set == ts)].iloc[0]
            vals.append((r.auroc, r.auroc_lo, r.auroc_hi))
        for j, (v, l, h) in enumerate(vals):
            d.errorbar(j + off, v, yerr=[[v - l], [h - v]], fmt="o" if j else "s",
                       mfc=col if j else "white", color=col, ms=6, capsize=3,
                       label=name if j == 1 else None)
            d.text(j + off + (0.06 if k else -0.06), v, f"{v:.3f}",
                   ha="left" if k else "right", va="center", fontsize=8, color=col)
    d.axhline(0.5, color=INK2, lw=1, ls=":")
    d.text(2.45, 0.505, "chance", fontsize=8, color=INK2, ha="right", va="bottom")
    d.set_xticks(range(3)); d.set_xticklabels(cats); d.set_xlim(-0.5, 2.5)
    d.set_ylim(0.45, 0.72); d.set_ylabel("AUROC")
    d.legend(fontsize=8, frameon=False, loc="upper right")
    d.set_title("D. Held up on the internal test, not on another population",
                loc="left", fontsize=10)
    d.text(0.02, 0.04, "squares: mean ± SD over 5 CV folds;\n"
                       "circles: 95% patient-level bootstrap CI",
           transform=d.transAxes, ha="left", va="bottom", fontsize=8, color=MUTED)

    for axx in ax.flat:
        axx.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.005, "Generated by src/training_evidence.py from results/ "
             "(tuning_d130_primary_*.csv, models.json, splits.json, metrics_overall.csv, "
             f"calibration_curves.csv). Patients in both training and test: "
             f"{splits['patient_overlap']}.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = C.FIGURES / "evidence_training.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
