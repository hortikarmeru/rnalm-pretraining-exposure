"""Masked-letter recovery gap by the size of the seen sequence's cluster in the RiNALMo corpus
(scripts 21, 30-32; config checks.masked_recovery.cluster_size_bins).

RiNALMo draws one sequence per cluster in each epoch, so a member of a large cluster is rarely drawn,
while models trained on the deduplicated corpus without clustering (RNA-FM, AIDO.RNA) see every member
of a dense neighbourhood every epoch. This reports each model's gap with pairs split by the seen
sequence's cluster size (clusters from script 21). Each bin uses script 32's test (family-weighted
estimate, family bootstrap, family sign flips) on the loss and accuracy gaps. Pairs whose seen sequence
id is not in clusters.tsv (the corpus kept another copy's header) are left out; the output counts them.

Usage: python scripts/33_recovery_by_cluster_size.py   (after scripts 21 and 31; environment-extra.yml)

Writes results/33_recovery_by_cluster_size.json plus its .meta.json.
"""
import importlib.util
import json

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta


def load_script32():
    spec = importlib.util.spec_from_file_location("recovery_analysis", repo_path("scripts/32_recovery_analysis.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cluster_sizes(path, wanted):
    """Size of the cluster each wanted sequence belongs to. clusters.tsv: representative, member."""
    rep = {}
    with open(path) as f:   # two passes, so only the wanted sequences' clusters are held
        for line in f:
            r, m = line.rstrip("\n").split("\t")
            if m in wanted:
                rep[m] = r
    reps = set(rep.values())
    size = dict.fromkeys(reps, 0)
    with open(path) as f:
        for line in f:
            r = line.split("\t", 1)[0]
            if r in reps:
                size[r] += 1
    return {m: size[r] for m, r in rep.items()}


def main():
    config = load_config()
    paths, cfg = config["paths"], config["checks"]["masked_recovery"]
    results = repo_path(paths["results_dir"])
    seed = config["random_seed"]
    s32 = load_script32()
    by_pair = {}
    for r in s32.read_tsv(results / "30_recovery_sequences.tsv"):
        by_pair.setdefault(r["pair"], {})[r["side"]] = r
    family = {p: d["seen"]["family"] for p, d in by_pair.items()}

    clusters = repo_path(paths["exposure_dir"].format(processed_dir=paths["processed_dir"])) / "rinalmo" / "clusters.tsv"
    seen = {d["seen"]["seq_id"]: p for p, d in by_pair.items()}
    size = {seen[q]: n for q, n in cluster_sizes(clusters, set(seen)).items()}
    edges = cfg["cluster_size_bins"]   # lower edges of the bins after the first
    bounds = list(zip([1] + edges, [e - 1 for e in edges] + [None]))
    bins = {(f"{lo}_to_{hi}" if hi else f"{lo}_and_above"): {p for p, n in size.items() if n >= lo and (hi is None or n <= hi)}
            for lo, hi in bounds}

    out = {"pairs_with_cluster": len(size), "pairs": len(by_pair),
           "bins": {b: {"pairs": len(ps), "families": len({family[p] for p in ps})} for b, ps in bins.items()},
           "models": {}}
    for key in cfg["models"]:
        rec = {r["seq_id"]: r for r in s32.read_tsv(results / f"31_recovery_{key}.tsv")}
        acc = {q: int(r["correct"]) / int(r["masked"]) for q, r in rec.items()}
        loss = {q: float(r["loss_sum"]) / int(r["masked"]) for q, r in rec.items()}
        gaps = {"loss": {p: loss[d["unseen"]["seq_id"]] - loss[d["seen"]["seq_id"]] for p, d in by_pair.items()},
                "accuracy": {p: acc[d["seen"]["seq_id"]] - acc[d["unseen"]["seq_id"]] for p, d in by_pair.items()}}
        out["models"][key] = {m: {b: s32.test({p: v[p] for p in ps}, family, cfg, seed) for b, ps in bins.items()}
                              for m, v in gaps.items()}
    path = results / "33_recovery_by_cluster_size.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["bins"]))
    for key in cfg["models"]:
        for b, t in out["models"][key][cfg["primary_metric"]].items():
            print(key, b, {k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()}, flush=True)
    write_meta(path, config, packages=["numpy"])


if __name__ == "__main__":
    main()
