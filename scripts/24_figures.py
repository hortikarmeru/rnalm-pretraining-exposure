"""Figures for the paper, from committed results (scripts 19 and 23) and script 19's analysis tables.

Usage: python scripts/24_figures.py   (environment-extra.yml)

Writes to paths.figures_dir, each as .png and .pdf:
  exposure_slopes     main exposure slope per model (F1 per 10x homologs) with 95% CI; filled marker =
                      within-family permutation p < 0.05
  exposure_curves     GAMM curve of the exposure effect per model (script 23), log scale, with +-2 SE bands
  exposure_cloud_<m>  every test sequence's F1 against homologs, per model, with the median per count bin
"""
import json
import math

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rnalm_exposure.config import load_config, repo_path

LABELS = {"rinalmo_micro": "RiNALMo micro", "rinalmo_mega": "RiNALMo mega", "rinalmo_giga": "RiNALMo giga",
          "rna_fm": "RNA-FM", "aido_rna_650m": "AIDO.RNA 650M", "aido_rna_1.6b": "AIDO.RNA 1.6B"}
ORDER = list(LABELS)
LN10 = math.log(10)


def save(fig, figures, name):
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def log_axis(ax, top):
    ticks = [t for t in [0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6] if math.log1p(t) <= top + 0.5]
    ax.set_xticks([math.log1p(t) for t in ticks])
    ax.set_xticklabels([f"{int(t):,}" if t < 1e3 else f"{int(t / 1e3)}k" if t < 1e6 else "1M" for t in ticks])
    ax.set_xlabel("Homologs in the training corpus (log scale)")


def main():
    config = load_config()
    results, figures = repo_path(config["paths"]["results_dir"]), repo_path(config["paths"]["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((results / "19_analysis_real.json").read_text())["models"]
    shape = json.loads((results / "23_shape_checks.json").read_text())
    models = [m for m in ORDER if m in analysis]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, m in enumerate(models):
        t, p = analysis[m]["main"]["terms"]["log_homologs"], analysis[m]["main"]["permutation"]["log_homologs"]["p"]
        y = len(models) - 1 - i
        ax.plot([t["ci_low"] * LN10, t["ci_high"] * LN10], [y, y], color="0.45", lw=2)
        ax.plot(t["estimate"] * LN10, y, "o", ms=8, color="0.45", mfc="0.45" if p < 0.05 else "white", mew=2)
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LABELS[m] for m in reversed(models)])
    ax.set_xlabel("F1 change per 10x more homologs (95% CI)")
    ax.set_title("Exposure effect per model; filled = within-family permutation p < 0.05", fontsize=10)
    save(fig, figures, "exposure_slopes")

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
    for ax, m in zip(axes.flat, models):
        c = pd.DataFrame(shape[m]["gamm"]["curve"])
        x = np.log1p(c["homologs"])
        ax.fill_between(x, c["gain"] - 2 * c["se"], c["gain"] + 2 * c["se"], color="tab:blue", alpha=0.15)
        ax.plot(x, c["gain"], color="tab:blue", lw=2)
        ax.axhline(0, color="0.2", lw=0.8)
        ax.set_title(LABELS[m], fontsize=10)
        log_axis(ax, x.max())
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 change vs 0 homologs")
    fig.suptitle("Exposure effect without a straight-line assumption (GAMM, family random effects)", fontsize=11)
    fig.tight_layout()
    save(fig, figures, "exposure_curves")

    table_dir = repo_path(config["paths"]["processed_dir"]) / "analysis" / "real"
    edges = config["analysis"]["shape_checks"]["bins"]
    for m in models:
        d = pd.read_csv(table_dir / f"{m}.tsv", sep="\t")
        n = np.round(np.expm1(d["log_homologs"]))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(d["log_homologs"], d["f1_primary"], s=4, color="0.6", alpha=0.3, lw=0)
        for lo, hi in zip(edges, edges[1:] + [math.inf]):
            k = (n >= lo) & (n < hi)
            if k.sum() < 20:
                continue
            x = math.log1p(lo) if hi - lo == 1 else math.log1p(math.sqrt(max(lo, 1) * min(hi - 1, n[k].max())))
            q1, med, q3 = np.quantile(d["f1_primary"][k], [0.25, 0.5, 0.75])
            ax.plot([x, x], [q1, q3], color="tab:blue", lw=2.5)
            ax.plot(x, med, "o", color="tab:blue", ms=7)
        log_axis(ax, d["log_homologs"].max())
        ax.set_ylabel("Test-set F1")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{LABELS[m]}: every test sequence; blue = median per range (middle half)", fontsize=10)
        save(fig, figures, f"exposure_cloud_{m}")


if __name__ == "__main__":
    main()
