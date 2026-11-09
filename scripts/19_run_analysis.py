"""Build the per-model analysis tables and fit the mixed models (config analysis and checks) with
scripts/19_fit_models.R (lme4 + lmerTest).

Usage: python scripts/19_run_analysis.py (--real | --simulated) [--permutations N]
  --real       F1 from script 18 (results/18_test_scores_<model>.tsv); models without scores are skipped.
  --simulated  F1 generated from the real covariates (config analysis.simulation), to test the whole
               analysis before any test scoring. Only F1 is synthetic.
  --permutations N  overrides checks.permutation.n_permutations (for a quick run of the simulation).

One row per test sequence and model: dataset, family (Rfam accession from script 15; sequences without
a family form their own group), length, F1 per readout, log_homologs in the model's corpus (script 13)
and vienna_f1 (script 14). The DNA floor has no exposure and is not fitted. When script 13 has run on
the post_snapshot corpus (placebo, every model) or the next_release corpus (RNA-FM), their counts are
added as log_placebo / log_next_release and fitted in place of log_homologs.

Writes <processed_dir>/analysis/<real|simulated>/*.tsv (tables) and results/19_analysis_<real|simulated>.json
(every fit, the fallback step it reached, the exposure and size terms, permutation p-values) with its
.meta.json (R session info included).
"""
import csv
import json
import math
import subprocess
import sys

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta

# The formulas scripts/19_fit_models.R fits; main() stops if the config's differ, so the two can't drift.
FORMULA = "f1 ~ log_homologs + vienna_f1 + log_length + dataset + (1 + log_homologs | family)"
JOINT_FORMULA = ("f1 ~ log_homologs * size + vienna_f1 + log_length + dataset + (1 + log_homologs | family)"
                 " + (1 | seq_id)")
EXTRA_EXPOSURE = {"next_release": "next_release", "placebo": "post_snapshot"}   # column -> corpus (script 13)


def read_tsv(path, key="seq_id"):
    with open(path) as f:
        return {r[key]: r for r in csv.DictReader(f, delimiter="\t")}


def covariates(config, corpus, key):
    paths, results = config["paths"], repo_path(config["paths"]["results_dir"])
    with open(repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))) as f:
        tests = [json.loads(line) for line in f]
    families = read_tsv(results / "15_families.tsv")
    vienna = read_tsv(results / "14_vienna.tsv")
    exposure = read_tsv(results / f"13_exposure_{corpus}.tsv")
    # next release: RNA-FM only (its snapshot is the uncertain one); placebo: every model
    extra = {col: read_tsv(results / f"13_exposure_{c}.tsv") for col, c in EXTRA_EXPOSURE.items()
             if (results / f"13_exposure_{c}.tsv").exists() and (col == "placebo" or key == "rna_fm")}
    rows = []
    for r in tests:
        row = {"seq_id": r["seq_id"], "dataset": r["dataset"],
               "family": families[r["seq_id"]]["family"] or r["seq_id"],   # no Rfam hit: its own group
               "length": len(r["sequence"]), "log_homologs": float(exposure[r["seq_id"]]["log_homologs"]),
               "vienna_f1": float(vienna[r["seq_id"]]["vienna_f1"])}
        for col in EXTRA_EXPOSURE:
            row[f"log_{col}"] = float(extra[col][r["seq_id"]]["log_homologs"]) if col in extra else "NA"
        rows.append(row)
    return rows


def simulate(rows_by_model, config, readouts):
    """F1 = intercept + exposure and vienna slopes + family intercept and slope + sequence effect
    (shared across one model family's sizes) + noise, clipped to [0, 1]. Covariates are real."""
    sim = config["analysis"]["simulation"]
    rng = np.random.default_rng(config["random_seed"])
    families = sorted({r["family"] for rows in rows_by_model.values() for r in rows})
    fam_int = dict(zip(families, rng.normal(0, sim["family_sd"], len(families))))
    fam_slope = dict(zip(families, rng.normal(0, sim["family_slope_sd"], len(families))))
    seq_ids = sorted({r["seq_id"] for rows in rows_by_model.values() for r in rows})
    seq_eff = dict(zip(seq_ids, rng.normal(0, sim["sequence_sd"], len(seq_ids))))
    for key, rows in sorted(rows_by_model.items()):
        mean_x = np.mean([r["log_homologs"] for r in rows])
        mean_v = np.mean([r["vienna_f1"] for r in rows])
        for r in rows:
            x, v = r["log_homologs"] - mean_x, r["vienna_f1"] - mean_v
            base = sim["intercept"] + (sim["exposure_slope"] + fam_slope[r["family"]]) * x + sim["vienna_slope"] * v \
                + fam_int[r["family"]] + seq_eff[r["seq_id"]]
            for readout in readouts:
                r[f"f1_{readout}"] = float(np.clip(base + rng.normal(0, sim["residual_sd"]), 0, 1))


def real_f1(rows_by_model, config, readouts):
    results = repo_path(config["paths"]["results_dir"])
    for key in list(rows_by_model):
        path = results / f"18_test_scores_{key}.tsv"
        if not path.exists():
            print(f"{key}: no test scores, skipped", flush=True)
            del rows_by_model[key]
            continue
        scores = {}
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                scores[(r["seq_id"], r["readout"])] = float(r["f1"])
        for r in rows_by_model[key]:
            for readout in readouts:
                r[f"f1_{readout}"] = scores[(r["seq_id"], readout)]


def main():
    config = load_config()
    paths, analysis, checks = config["paths"], config["analysis"], config["checks"]
    mode = {"--real": "real", "--simulated": "simulated"}
    modes = [mode[a] for a in sys.argv[1:] if a in mode]
    if len(modes) != 1:
        raise SystemExit("pass exactly one of --real or --simulated")
    tag = modes[0]
    if analysis["formula"] != FORMULA or analysis["rinalmo_joint_formula"] != JOINT_FORMULA:
        raise ValueError("config analysis formulas differ from the ones scripts/19_fit_models.R fits")
    readouts = ["primary", *config["readout"]["variants"]]
    n_perm = checks["permutation"]["n_permutations"]
    if "--permutations" in sys.argv:
        n_perm = int(sys.argv[sys.argv.index("--permutations") + 1])

    corpora = config["validation"]["exposure_corpus"]
    rows_by_model = {}
    for key, corpus in corpora.items():
        if (repo_path(paths["results_dir"]) / f"13_exposure_{corpus}.tsv").exists():
            rows_by_model[key] = covariates(config, corpus, key)
        else:
            print(f"{key}: no exposure table for corpus {corpus}, skipped", flush=True)
    (simulate if tag == "simulated" else real_f1)(rows_by_model, config, readouts)

    table_dir = repo_path(paths["processed_dir"]) / "analysis" / tag
    table_dir.mkdir(parents=True, exist_ok=True)
    columns = ["seq_id", "dataset", "family", "length", "log_homologs", "vienna_f1",
               *(f"log_{c}" for c in EXTRA_EXPOSURE), *(f"f1_{r}" for r in readouts)]
    for key, rows in rows_by_model.items():
        with open(table_dir / f"{key}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, columns, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    settings = {
        "table_dir": str(table_dir), "models": sorted(rows_by_model),
        "readouts": readouts,
        "max_placebo_correlation": checks["placebo"]["max_correlation_with_exposure"],
        "joint_models": {name: [k for k in keys if k in rows_by_model]
                         for name, keys in analysis["joint_models"].items()},
        "log10_parameters": {k: math.log10(v) for k, v in analysis["parameters"].items()},
        "min_length": analysis["sensitivity_min_length"],
        "n_permutations": n_perm, "seed": checks["permutation"]["seed"],
        "out": str(repo_path(paths["results_dir"]) / f"19_analysis_{tag}.json"),
    }
    settings_path = table_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))
    subprocess.run(["Rscript", str(repo_path("scripts/19_fit_models.R")), str(settings_path)], check=True)
    session = subprocess.run(["Rscript", "-e", "suppressMessages({library(lme4); library(lmerTest)}); sessionInfo()"],
                             capture_output=True, text=True).stdout
    write_meta(settings["out"], config, extra={"r_session": session, "permutations": n_perm})


if __name__ == "__main__":
    main()
