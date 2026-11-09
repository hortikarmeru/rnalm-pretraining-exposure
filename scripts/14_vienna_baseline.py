"""Fold every test sequence with RNAfold (ViennaRNA; settings in config vienna) and score its MFE
structure against the reference pairs with the same F1/MCC as the attention readout. vienna_f1 is
a covariate in the main regression: it stands for how foldable each sequence is.

Usage: python scripts/14_vienna_baseline.py

Writes results/14_vienna.tsv (seq_id, vienna_f1, vienna_mcc, mfe) and results/14_vienna.json
(summary per set) plus its .meta.json.
"""
import json
import os
import subprocess
import tempfile

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.metrics import score_pair_sets
from rnalm_exposure.structures import dot_bracket_pairs


def rnafold(records, cfg):
    """{seq_id: (dot-bracket, MFE)} from one RNAfold run over all records."""
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
        for r in records:
            f.write(f">{r['seq_id']}\n{r['sequence']}\n")
    try:
        out = subprocess.run(["RNAfold", "--noPS", "-T", str(cfg["temperature"]), f"--jobs={os.cpu_count()}",
                              *cfg["rnafold_options"], f.name], check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(f.name)
    lines = out.splitlines()
    folds = {}
    for k in range(0, len(lines), 3):   # >id, sequence, "structure ( mfe)"
        structure, energy = lines[k + 2].split(" ", 1)
        folds[lines[k][1:]] = (structure, float(energy.strip(" ()")))
    return folds


def main():
    config = load_config()
    paths = config["paths"]
    with open(repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))) as f:
        records = [json.loads(line) for line in f]
    folds = rnafold(records, config["vienna"])

    rows, per_set = [], {}
    for r in records:
        structure, mfe = folds[r["seq_id"]]
        if len(structure) != len(r["sequence"]):
            raise ValueError(f"{r['seq_id']}: RNAfold structure length differs from the sequence")
        score = score_pair_sets(dot_bracket_pairs(structure), {tuple(p) for p in r["pairs"]}, len(r["sequence"]))
        rows.append((r["seq_id"], score["f1"], score["mcc"], mfe))
        per_set.setdefault(r["dataset"], []).append(score["f1"])

    out_tsv = repo_path(paths["results_dir"]) / "14_vienna.tsv"
    with open(out_tsv, "w") as f:
        f.write("seq_id\tvienna_f1\tvienna_mcc\tmfe\n")
        for seq_id, f1, mcc, mfe in rows:
            f.write(f"{seq_id}\t{f1:.6f}\t{mcc:.6f}\t{mfe:.2f}\n")
    summary = {name: {"n": len(v), "median_f1": float(np.median(v)), "mean_f1": float(np.mean(v))}
               for name, v in per_set.items()}
    print(json.dumps(summary, indent=2))
    out = repo_path(paths["results_dir"]) / "14_vienna.json"
    with open(out, "w") as f:
        json.dump({"sequences": len(rows), "per_set": summary}, f, indent=2)
    rnafold_version = subprocess.run(["RNAfold", "--version"], capture_output=True, text=True).stdout.strip()
    write_meta(out, config, extra={"rnafold": rnafold_version})


if __name__ == "__main__":
    main()
