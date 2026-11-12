"""Shape of the exposure effect (config analysis.shape_checks): does the result of the main mixed
model hold without assuming a straight line in log exposure?

Usage: python scripts/23_shape_checks.py   (after scripts/19_run_analysis.py --real; environment-extra.yml)

On the per-model tables script 19 writes (primary readout), for every model with test scores:
  - scripts/23_shape_checks.R: a GAMM (smooth exposure curve, family random intercept and slope), exposure in
    count bins against sequences with exactly 1 homolog, and a beta mixed model (F1 is bounded in [0, 1]).
  - Gradient boosting with whole families held out (GroupKFold by family): R2 on unseen families without and
    with log_homologs among the features (vienna_f1, log_length, dataset), the mean gain over folds, a
    permutation test (log_homologs shuffled across sequences, p = (k + 1) / (n + 1)), and the partial
    dependence of predicted F1 on exposure.

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

PD_COUNTS = [0, 1, 9, 99, 999, 9999, 99999, 999999]


def boosting_gain(base, exposure, y, folds, cfg, seed, pd_grid=None):
    """Mean R2 gain on held-out families from adding exposure; optionally the partial dependence curve."""
    gains, curves = [], []
    for train, test in folds:
        r2 = {}
        for name, X in (("without", base), ("with", base.assign(log_homologs=exposure))):
            model = HistGradientBoostingRegressor(max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
                                                  random_state=seed).fit(X.iloc[train], y[train])
            r2[name] = r2_score(y[test], model.predict(X.iloc[test]))
            if name == "with" and pd_grid is not None:   # mean prediction on held-out rows with exposure set to each value
                Xt = X.iloc[test].copy()
                curves.append([float(model.predict(Xt.assign(log_homologs=v)).mean()) for v in pd_grid])
        gains.append((r2["without"], r2["with"]))
    return gains, curves


def boosting(table, cfg, seed):
    d = pd.read_csv(table, sep="\t")
    base = pd.get_dummies(d[["vienna_f1", "dataset"]].assign(log_length=np.log(d["length"])),
                          columns=["dataset"], dtype=float)
    y, exposure = d["f1_primary"].to_numpy(), d["log_homologs"].to_numpy()
    folds = list(GroupKFold(n_splits=cfg["folds"]).split(base, y, d["family"]))   # whole families held out
    grid = np.log1p([c for c in PD_COUNTS if np.log1p(c) <= exposure.max()])
    observed, curves = boosting_gain(base, exposure, y, folds, cfg, seed, grid)
    stat = float(np.mean([w - wo for wo, w in observed]))
    rng = np.random.default_rng(seed)
    shuffles = [rng.permutation(exposure) for _ in range(cfg["permutations"])]   # drawn in order: core count doesn't matter
    null = Parallel(n_jobs=os.cpu_count())(
        delayed(lambda e: float(np.mean([w - wo for wo, w in boosting_gain(base, e, y, folds, cfg, seed)[0]])))(e)
        for e in shuffles)
    k = int(sum(g >= stat for g in null))
    curve = np.mean(curves, axis=0)
    return {"r2_without": float(np.mean([wo for wo, _ in observed])), "r2_with": float(np.mean([w for _, w in observed])),
            "gain_by_fold": [float(w - wo) for wo, w in observed], "mean_gain": stat,
            "permutation": {"n": len(null), "k": k, "p": (k + 1) / (len(null) + 1),
                            "null_mean": float(np.mean(null)), "null_max": float(np.max(null))},
            "partial_dependence": {"homologs": [int(round(np.expm1(v))) for v in grid],
                                   "gain": [float(c - curve[0]) for c in curve]}}


def main():
    config = load_config()
    paths, cfg = config["paths"], config["analysis"]["shape_checks"]
    table_dir = repo_path(paths["processed_dir"]) / "analysis" / "real"
    models = sorted(p.stem for p in table_dir.glob("*.tsv"))
    out = repo_path(paths["results_dir"]) / "23_shape_checks.json"

    settings = {"table_dir": str(table_dir), "models": models, "gamm_k": cfg["gamm_k"], "bins": cfg["bins"],
                "beta_squeeze": cfg["beta_squeeze"], "out": str(out.with_suffix(".r.json"))}
    settings_path = table_dir / "shape_settings.json"
    settings_path.write_text(json.dumps(settings))
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    subprocess.run(["Rscript", str(repo_path("scripts/23_shape_checks.R")), str(settings_path)], check=True, env=env)
    results = json.loads(out.with_suffix(".r.json").read_text())
    out.with_suffix(".r.json").unlink()

    os.environ["OMP_NUM_THREADS"] = "1"   # one thread per boosting fit; the permutations run one per core
    for key in models:
        results[key]["boosting"] = boosting(table_dir / f"{key}.tsv", cfg["boosting"], config["random_seed"])
        b = results[key]["boosting"]
        print(key, round(b["r2_without"], 3), round(b["r2_with"], 3), "p =", round(b["permutation"]["p"], 4), flush=True)

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    session = subprocess.run(["Rscript", "-e", "suppressMessages({library(mgcv); library(glmmTMB); library(lme4)}); sessionInfo()"],
                             capture_output=True, text=True).stdout
    write_meta(out, config, packages=["scikit-learn", "numpy", "pandas", "joblib"], extra={"r_session": session})


if __name__ == "__main__":
    main()
