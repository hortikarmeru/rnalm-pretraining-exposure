"""The sequences every per-sequence search runs on: the test sequences (script 12) plus Wang et al.
2026's selection and reproduction sets, normalized (uppercase, T -> U)."""
import json

from rnalm_exposure.config import repo_path
from rnalm_exposure.test_sets import normalize
from rnalm_exposure.wang import load_set


def query_sequences(config):
    """{seq_id: sequence}, test sequences first."""
    paths, val = config["paths"], config["validation"]
    queries = {}
    with open(repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))) as f:
        for line in f:
            r = json.loads(line)
            queries[r["seq_id"]] = r["sequence"]
    zip_path = repo_path(val["wang2026_zip"].format(raw_dir=paths["raw_dir"]))
    for role in ("selection_set", "reproduction_set"):
        name = val[role]
        for r in load_set(zip_path, val["wang2026_files"][name], name):
            queries[r["seq_id"]] = normalize(r["sequence"])
    return queries


def write_fasta(seqs, path):
    with open(path, "w") as f:
        for seq_id, seq in seqs.items():
            f.write(f">{seq_id}\n{seq}\n")
