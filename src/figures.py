"""Step 6 - figures (PNG, 200 dpi) under figures/.

fig1_cohort_flow.png         exclusion flow for both cohorts
fig2_roc_pr.png              ROC and precision-recall curves, internal vs external
fig3_calibration_dca.png     reliability curves and decision curves (net benefit vs
                             treat-all / treat-none), internal vs external
fig4_shift_transfer.png      feature shift (SMD) and the transfer matrix
fig5_subgroups.png           gradient boosting: subgroup AUROC, TPR, FPR and
                             observed/expected ratio, internal vs external
figS1_recalibration.png      external calibration before / after recalibration
figS2_subgroups_lr.png       as fig5, logistic regression
figS3_risk_distribution.png  distributions of predicted risk, internal vs external
Figures are drawn at A4 print width (7.2 in), so fonts print at nominal size.
Usage:  python src/figures.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

import config as C  # noqa: E402
import harmonize as H  # noqa: E402

# palette (validated with the dataviz skill's validator: slots 1-3, all pairs pass)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                         "#c3c2b7", "#fcfcfb")
MODEL_STYLE = {"lr": (BLUE, "Logistic regression"), "hgb": (ORANGE, "Gradient boosting")}
SET_TITLE = {"d130_test": "Internal test (diabetes-130)",
             "synthea_all": "External (Synthea)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.labelsize": 8.5, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.axisbelow": True, "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
})


def load():
    preds = pd.read_parquet(C.RESULTS / "predictions.parquet")
    om = pd.read_csv(C.RESULTS / "metrics_overall.csv")
    return preds, om


def metric_label(om, mid, t, k, digits=3):
    r = om[(om.model_id == mid) & (om.test_set == t)].iloc[0]
    return f"{r[k]:.{digits}f} ({r[k + '_lo']:.{digits}f}-{r[k + '_hi']:.{digits}f})"


def save(fig, name):
    C.FIGURES.mkdir(exist_ok=True)
    fig.savefig(C.FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig]", name)


# ---------------------------------------------------------------------------
def fig_cohort_flow():
    d = json.loads((C.RESULTS / "cohort_flow_d130.json").read_text())
    s = json.loads((C.RESULTS / "cohort_flow_synthea.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_axis_off(); ax.set_xlim(0, 2); ax.set_ylim(0, 1)

    def box(x, y, text, w=0.9, h=0.1, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc="#ffffff", ec=AXIS, lw=0.8))
        ax.text(x, y, text, ha="center", va="center", fontsize=6.7,
                fontweight="bold" if bold else "normal", color=INK)

    def arrow(x, y0, y1):
        ax.annotate("", (x, y1), (x, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                                            lw=0.8))

    def short(step):
        step = step.replace("Excluded: ", "")
        for a, b in [(" (incomplete 365-day look-back)", ""), (" (incomplete follow-up)", ""),
                     (" (not a diabetic encounter)", ""), ("; Amendment 1)", ")"),
                     (" (discharge_disposition_id 11,13,14,19,20,21)", ""),
                     ("planned-care admission (primary diagnosis V57 rehabilitation or V58 "
                      "chemotherapy/aftercare)", "planned care (primary diagnosis)"),
                     ("planned-care admission (chemotherapy, rehabilitation/detox, transplant)",
                      "planned care (chemotherapy, rehab/detox, transplant)")]:
            step = step.replace(a, b)
        return step

    # diabetes-130 (left)
    x = 0.47
    box(x, 0.93, f"diabetes-130: {d[0]['encounters']:,} encounters\n"
                 f"{d[0]['patients']:,} patients", bold=True)
    lines = []
    for r in d[1:-1]:
        lines.append(f"-{r['excluded']:,} {short(r['step'])}")
        if "excluded_by_code" in r:
            lines.append("    (" + "; ".join(f"{k}: {v:,}" for k, v in
                                           r["excluded_by_code"].items()) + ")")
    box(x, 0.52, "\n".join(lines), h=0.36, w=0.93)
    fin = d[-1]
    box(x, 0.1, f"Analysis cohort: {fin['encounters']:,} encounters\n"
                f"{fin['patients']:,} patients | readmitted <30 d: "
                f"{fin['readmit_30d']:,} ({100 * fin['prevalence']:.1f}%)", bold=True, h=0.13)
    arrow(x, 0.88, 0.71); arrow(x, 0.34, 0.17)
    # Synthea (right)
    x = 1.52
    box(x, 0.93, f"Synthea: {s[0]['patients']:,} patients with type-2 diabetes\n"
                 f"{s[1]['inpatient_encounters']:,} inpatient encounters -> "
                 f"{s[1]['stays']:,} merged stays", bold=True)
    lines = [f"-{r['excluded']:,} {short(r['step'])}" for r in s[2:-1]]
    box(x, 0.52, "\n".join(lines), h=0.36, w=0.93)
    fin = s[-1]
    box(x, 0.1, f"Analysis cohort: {fin['stays']:,} stays\n{fin['patients']:,} patients | "
                f"readmitted <30 d: {fin['readmit_30d']:,} ({100 * fin['prevalence']:.1f}%)",
        bold=True, h=0.13)
    arrow(x, 0.88, 0.71); arrow(x, 0.34, 0.17)
    save(fig, "fig1_cohort_flow.png")


def _strip_legend(fig, extra=()):
    """One shared legend above a strip of panels."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=1.6, label=n) for c, n in MODEL_STYLE.values()]
    handles += [Line2D([], [], color=c, lw=ls_w, ls="-", label=n) for c, ls_w, n in extra]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.06), fontsize=7)


def fig_roc_pr(preds, om):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for j, t in enumerate(["d130_test", "synthea_all"]):
        ax_roc, ax_pr = axes[j], axes[2 + j]
        for kind in ["lr", "hgb"]:
            mid = f"d130|primary|{kind}"
            g = preds[(preds.model_id == mid) & (preds.test_set == t)]
            col, _ = MODEL_STYLE[kind]
            fpr, tpr, _ = roc_curve(g.y, g.p)
            ax_roc.plot(fpr, tpr, color=col, lw=1.4)
            prec, rec, _ = precision_recall_curve(g.y, g.p)
            ax_pr.plot(rec, prec, color=col, lw=1.4)
        prev = g.y.mean()
        ax_roc.plot([0, 1], [0, 1], color=MUTED, lw=0.9)
        ax_pr.axhline(prev, color=MUTED, lw=0.9)
        where = "internal" if t == "d130_test" else "external"
        ax_roc.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="1 - specificity",
                   ylabel="Sensitivity" if j == 0 else None)
        ax_pr.set(xlim=(0, 1), ylim=(0, max(0.5, prev * 3)), xlabel="Recall",
                  ylabel="Precision" if j == 0 else None)
        ax_roc.set_title(f"ROC, {where}", fontsize=8)
        ax_pr.set_title(f"Precision-recall, {where}", fontsize=8)
        for ax in (ax_roc, ax_pr):
            ax.tick_params(labelsize=6.5)
            ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
    _strip_legend(fig, extra=[(MUTED, 0.9, "Chance / prevalence")])
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2_roc_pr.png")


def fig_calibration_dca(om):
    cal = pd.read_csv(C.RESULTS / "calibration_curves.csv")
    dca = pd.read_csv(C.RESULTS / "dca.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for j, t in enumerate(["d130_test", "synthea_all"]):
        ax, axd = axes[j], axes[2 + j]
        where = "internal" if t == "d130_test" else "external"
        hi = 0.0
        for kind in ["lr", "hgb"]:
            mid = f"d130|primary|{kind}"
            col, _ = MODEL_STYLE[kind]
            c = cal[(cal.model_id == mid) & (cal.test_set == t)]
            ax.errorbar(c.mean_pred, c.observed,
                        yerr=[c.observed - c.obs_lo, c.obs_hi - c.observed], color=col,
                        marker="o", ms=3.6, mec=SURFACE, mew=0.8, lw=1.2, capsize=0,
                        elinewidth=0.8)
            hi = max(hi, c.mean_pred.max(), c.obs_hi.max())
        lim = min(1, hi * 1.08)
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.9)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted risk",
               ylabel="Observed rate" if j == 0 else None)
        ax.set_title(f"Calibration, {where}", fontsize=8)
        base = dca[(dca.test_set == t) & (dca.model_id == "d130|primary|lr")]
        axd.plot(base.threshold, base.nb_all, color=INK2, lw=0.9)
        axd.axhline(0, color=MUTED, lw=0.9)
        top = max(base.nb_all.max(), 0)
        for kind in ["lr", "hgb"]:
            mid = f"d130|primary|{kind}"
            col, _ = MODEL_STYLE[kind]
            d = dca[(dca.test_set == t) & (dca.model_id == mid)]
            axd.fill_between(d.threshold, d.nb_model_lo, d.nb_model_hi, color=col,
                             alpha=0.12, lw=0)
            axd.plot(d.threshold, d.nb_model, color=col, lw=1.3)
            top = max(top, d.nb_model_hi.max())
        axd.set(xlim=(0, 0.4), ylim=(-0.01, top * 1.15 + 0.002), xlabel="Threshold",
                ylabel="Net benefit" if j == 0 else None)
        axd.set_title(f"Decision curve, {where}", fontsize=8)
        for a in (ax, axd):
            a.tick_params(labelsize=6.5)
            a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
    _strip_legend(fig, extra=[(MUTED, 0.9, "Ideal calibration / treat none"),
                              (INK2, 0.9, "Treat all")])
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig3_calibration_dca.png")


def fig_risk_distribution():
    preds = pd.read_parquet(C.RESULTS / "predictions.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for j, kind in enumerate(["lr", "hgb"]):
        ax = axes[j]
        mid = f"d130|primary|{kind}"
        for t, col, name in [("d130_test", BLUE, "Internal test"),
                             ("synthea_all", ORANGE, "External (Synthea)")]:
            g = preds[(preds.model_id == mid) & (preds.test_set == t)]
            ax.hist(g.p, bins=np.linspace(0, float(preds.p.quantile(0.995)), 45),
                    histtype="step", color=col, lw=1.3, density=True, label=name)
        ax.set(xlabel="Predicted risk", ylabel="Density")
        ax.set_title(MODEL_STYLE[kind][1], fontsize=8.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "figS3_risk_distribution.png")


def _subgroup_rows(fg, attr_order=("race", "sex", "age_band", "insurance")):
    """y positions for a forest plot: attribute headers + groups."""
    order = {"race": H.RACE_ORDER, "sex": H.SEX_ORDER, "age_band": H.AGE_BAND_ORDER,
             "insurance": H.INSURANCE_ORDER}
    names = {"race": "Race", "sex": "Sex", "age_band": "Age band", "insurance": "Insurance"}
    labels, pos = [], {}
    y = 0
    for a in attr_order:
        present = set(fg[fg.attribute == a].group)
        labels.append((y, names[a], True)); y += 1
        for g in order[a]:
            if g in present:
                pos[(a, g)] = y; labels.append((y, g, False)); y += 1
        y += 0.4
    return pos, labels, y


SET_STYLE = {"d130_test": (BLUE, "o", "Internal (diabetes-130 test)"),
             "synthea_all": (ORANGE, "s", "External (Synthea)")}


def _forest(ax, fg, mid, metric, pos, xlabel, ref=None):
    """Point + CI per subgroup; internal and external offset vertically."""
    for t, (col, marker, name) in SET_STYLE.items():
        d = fg[(fg.model_id == mid) & (fg.test_set == t)]
        labelled = False
        for r in d.itertuples():
            y = pos[(r.attribute, r.group)] + (-0.17 if t == "d130_test" else 0.17)
            v = getattr(r, metric)
            if not r.reportable or v != v:
                continue
            lo, hi = getattr(r, metric + "_lo"), getattr(r, metric + "_hi")
            ax.plot([lo, hi], [y, y], color=col, lw=1.1)
            ax.plot(v, y, marker, color=col, ms=3.8, mec=SURFACE, mew=0.8,
                    label=None if labelled else name)
            labelled = True
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=0.9)
    ax.set_xlabel(xlabel, fontsize=7)


def fig_subgroups(kind="hgb", name="fig5_subgroups.png"):
    fg = pd.read_csv(C.RESULTS / "fairness_groups.csv")
    mid = f"d130|primary|{kind}"
    pos, labels, ymax = _subgroup_rows(fg)
    cols = [("auroc", "AUROC", 0.5), ("tpr", "TPR (sensitivity)", None),
            ("fpr", "FPR", None), ("oe_ratio", "Observed / expected", 1.0)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.45), sharey=True)
    for j, (metric, xl, ref) in enumerate(cols):
        _forest(axes[j], fg, mid, metric, pos, xl, ref=ref)
        axes[j].set_title(xl.split(" (")[0], fontsize=8)
        axes[j].tick_params(axis="x", labelsize=6.5)
    axes[0].set_yticks([p for p, _, _ in labels])
    axes[0].set_yticklabels([("" if h else "  ") + s_ for _, s_, h in labels], fontsize=7)
    for tl, (_, _, h) in zip(axes[0].get_yticklabels(), labels):
        if h:
            tl.set_fontweight("bold"); tl.set_color(INK)
    axes[0].set_ylim(ymax, -0.8)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.965),
               fontsize=7.2, title=MODEL_STYLE[kind][1] + ":", title_fontsize=7.2,
               alignment="left")
    fig.text(0.99, 0.003, "95% CIs. Gray lines: AUROC 0.5 and O/E 1. TPR/FPR at the "
             f"pre-specified threshold. Groups with <{C.MIN_SUBGROUP_N} stays or "
             f"<{C.MIN_SUBGROUP_EVENTS} events are not shown.",
             ha="right", fontsize=6.1, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955), w_pad=0.6)
    save(fig, name)


def fig_shift_transfer():
    smd = pd.read_csv(C.RESULTS / "shift_smd.csv")
    om = pd.read_csv(C.RESULTS / "metrics_overall.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    smd = smd.assign(a=smd.smd.abs()).sort_values("a")
    y = np.arange(len(smd))
    ax.plot(smd.smd.clip(-3, 3), y, "o", color=BLUE, ms=4.5, mec=SURFACE, mew=1.0)
    for v in (-0.1, 0.1):
        ax.axvline(v, color=MUTED, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(smd.label, fontsize=6.6)
    ax.set_xlabel("Standardized mean difference (Synthea - diabetes-130)\n"
                  "values beyond +/-3 drawn at the edge")
    ax.set_title("Feature shift between cohorts", fontsize=8.5)
    ax = axes[1]
    combos = [("d130", "d130_test", "diabetes-130 -> diabetes-130"),
              ("d130", "synthea_test", "diabetes-130 -> Synthea"),
              ("synthea", "synthea_test", "Synthea -> Synthea"),
              ("synthea", "d130_test", "Synthea -> diabetes-130")]
    for i, (src, t, _) in enumerate(combos):
        for kind in ["lr", "hgb"]:
            col, name = MODEL_STYLE[kind]
            r = om[(om.model_id == f"{src}|primary|{kind}") & (om.test_set == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = i + (-0.12 if kind == "lr" else 0.12)
            ax.plot([r.auroc_lo, r.auroc_hi], [yy, yy], color=col, lw=1.2)
            ax.plot(r.auroc, yy, "o", color=col, ms=4.5, mec=SURFACE, mew=1.0,
                    label=name if i == 0 else None)
    ax.axvline(0.5, color=MUTED, lw=0.9)
    ax.set_yticks(range(len(combos))); ax.set_yticklabels([c[2] for c in combos], fontsize=7)
    ax.set_ylim(len(combos) - 0.5, -0.5)
    ax.set_xlabel("AUROC (95% CI) on the 20% test split")
    ax.set_title("Transfer matrix (train -> test)", fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig4_shift_transfer.png")


def fig_recalibration():
    cal = pd.read_csv(C.RESULTS / "calibration_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for j, kind in enumerate(["lr", "hgb"]):
        ax = axes[j]
        mid = f"d130|primary|{kind}"
        hi = 0
        for suffix, col, name in [("", MUTED, "Original"),
                                  ("|recal_intercept", BLUE, "Intercept recalibrated"),
                                  ("|recal_logistic", ORANGE, "Intercept + slope")]:
            c = cal[(cal.model_id == mid + suffix) & (cal.test_set == "synthea_half_B")]
            ax.errorbar(c.mean_pred, c.observed, yerr=[c.observed - c.obs_lo,
                                                       c.obs_hi - c.observed],
                        color=col, marker="o", ms=4, mec=SURFACE, mew=1.0, lw=1.3,
                        elinewidth=0.8, label=name)
            hi = max(hi, c.mean_pred.max(), c.obs_hi.max())
        lim = min(1, hi * 1.08)
        ax.plot([0, lim], [0, lim], color=AXIS, lw=0.9)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted risk (decile)",
               ylabel="Observed readmission")
        ax.set_title(f"{MODEL_STYLE[kind][1]} on Synthea half B", fontsize=8.5)
        ax.legend(loc="upper left", fontsize=6.8)
    fig.tight_layout()
    save(fig, "figS1_recalibration.png")


def main():
    preds, om = load()
    for old in C.FIGURES.glob("fig*.png"):  # drop figures from earlier layouts
        old.unlink()
    fig_cohort_flow()
    fig_roc_pr(preds, om)
    fig_calibration_dca(om)
    fig_shift_transfer()
    fig_subgroups("hgb", "fig5_subgroups.png")
    fig_recalibration()
    fig_subgroups("lr", "figS2_subgroups_lr.png")
    fig_risk_distribution()


if __name__ == "__main__":
    main()
