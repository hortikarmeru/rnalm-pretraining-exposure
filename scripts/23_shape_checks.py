"""Shape of the exposure effect (config analysis.shape_checks): does the result of the main mixed
model hold without assuming a straight line in log exposure?

Usage: python scripts/23_shape_checks.py   (after scripts/19_run_analysis.py --real; environment-extra.yml)

On the per-model tables script 19 writes (primary readout), for every model with test scores:
  - scripts/23_shape_checks.R: a GAMM (smooth exposure curve, family random intercept and slope), exposure in
    count bins against sequences with exactly 1 homolog, and a beta mixed model (F1 is bounded in [0, 1]).
  - Gradient boosting with whole families held out (GroupKFold by family): R2 on unseen families without and
    with log_homologs among the features (vienna_f1, log_length, dataset), the mean gain over folds, and the
    partial dependence of predicted F1 on exposure. Two permutation nulls: log_homologs shuffled within each
    family (as script 19's check; information within families) and across all sequences (whether exposure,
    including its family-level part, adds information). p = (k + 1) / (n + 1).

Writes results/23_shape_checks.json with its .meta.json.
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta

def fit_r2(X, y, train, test, cfg, seed):
    model = HistGradientBoostingRegressor(max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
                                          early_stopping=False, random_state=seed).fit(X.iloc[train], y[train])
    return model, r2_score(y[test], model.predict(X.iloc[test]))


def r2_with(base, exposure, y, folds, cfg, seed, pd_grid=None):
    """R2 on each held-out fold with exposure among the features; optionally the partial dependence curve."""
    r2s, curves = [], []
    X = base.assign(log_homologs=exposure)
    for train, test in folds:
        model, r2 = fit_r2(X, y, train, test, cfg, seed)
        r2s.append(r2)
        if pd_grid is not None:   # mean prediction on held-out rows with exposure set to each value
            curves.append([float(model.predict(X.iloc[test].assign(log_homologs=v)).mean()) for v in pd_grid])
    return r2s, curves


def within_family(exposure, families, rng):
    out = exposure.copy()
    for idx in pd.Series(range(len(families))).groupby(families).indices.values():
        if len(idx) > 1:
            out[idx] = exposure[rng.permutation(idx)]
    return out


def boosting(table, cfg, seed, pd_counts):
    d = pd.read_csv(table, sep="\t")
    base = pd.get_dummies(d[["vienna_f1", "dataset"]].assign(log_length=np.log(d["length"])),
                          columns=["dataset"], dtype=float)
    y, exposure = d["f1_primary"].to_numpy(), d["log_homologs"].to_numpy()
    folds = list(GroupKFold(n_splits=cfg["folds"]).split(base, y, d["family"]))   # whole families held out
    grid = np.log1p([c for c in pd_counts if np.log1p(c) <= exposure.max()])
    without = [fit_r2(base, y, tr, te, cfg, seed)[1] for tr, te in folds]   # the same for every permutation
    observed, curves = r2_with(base, exposure, y, folds, cfg, seed, grid)
    gain = lambda r2s: float(np.mean(np.subtract(r2s, without)))
    stat = gain(observed)
    families = d["family"].to_numpy()
    rng = np.random.default_rng(seed)   # shuffles drawn in order, so the core count doesn't change them
    nulls = {"within_family": [within_family(exposure, families, rng) for _ in range(cfg["permutations"])],
             "global": [rng.permutation(exposure) for _ in range(cfg["permutations"])]}
    perm = {}
    for name, shuffles in nulls.items():
        null = Parallel(n_jobs=cfg["n_jobs"])(delayed(lambda e: gain(r2_with(base, e, y, folds, cfg, seed)[0]))(e)
                                             for e in shuffles)
        k = int(sum(g >= stat for g in null))
        perm[name] = {"n": len(null), "k": k, "p": (k + 1) / (len(null) + 1),
                      "null_mean": float(np.mean(null)), "null_max": float(np.max(null))}
    curve = np.mean(curves, axis=0)
    return {"r2_without": float(np.mean(without)), "r2_with": float(np.mean(observed)),
            "gain_by_fold": [float(w - wo) for wo, w in zip(without, observed)], "mean_gain": stat,
            "permutation": perm,
            "partial_dependence": {"homologs": [int(round(np.expm1(v))) for v in grid],
                                   "gain": [float(c - curve[0]) for c in curve]}}


def main():
    config = load_config()
    paths, cfg = config["paths"], config["analysis"]["shape_checks"]
    table_dir = repo_path(paths["processed_dir"]) / "analysis" / "real"
    models = sorted(p.stem for p in table_dir.glob("*.tsv"))
    out = repo_path(paths["results_dir"]) / "23_shape_checks.json"

    settings = {"table_dir": str(table_dir), "models": models, "gamm_k": cfg["gamm_k"], "bins": cfg["bins"],
                "beta_squeeze": cfg["beta_squeeze"], "curve_counts": cfg["curve_counts"], "out": str(out.with_suffix(".r.json"))}
    settings_path = table_dir / "shape_settings.json"
    settings_path.write_text(json.dumps(settings))
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    subprocess.run(["Rscript", str(repo_path("scripts/23_shape_checks.R")), str(settings_path)], check=True, env=env)
    results = json.loads(out.with_suffix(".r.json").read_text())
    out.with_suffix(".r.json").unlink()

    os.environ["OMP_NUM_THREADS"] = "1"   # one thread per boosting fit; the permutations run one per core
    for key in models:
        results[key]["boosting"] = boosting(table_dir / f"{key}.tsv", cfg["boosting"], config["random_seed"], cfg["curve_counts"])
        b = results[key]["boosting"]
        print(key, round(b["r2_without"], 3), round(b["r2_with"], 3),
              {k: round(v["p"], 4) for k, v in b["permutation"].items()}, flush=True)

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    session = subprocess.run(["Rscript", "-e", "suppressMessages({library(mgcv); library(glmmTMB); library(lme4)}); sessionInfo()"],
                             capture_output=True, text=True).stdout
    write_meta(out, config, packages=["scikit-learn", "numpy", "pandas", "joblib"], extra={"r_session": session})


if __name__ == "__main__":
    main()
