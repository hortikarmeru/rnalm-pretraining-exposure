"""Build the four test sets: read each structure file from its downloaded archive, normalize the
sequence, drop sequences with letters other than ACGU or over the length cutoff, and resolve exact
duplicates within and across sets (rules and their reasons in config test_sets).

Usage: python scripts/12_prepare_test_sets.py

Writes paths.test_sets (one JSON record per line: seq_id, dataset, file, sequence, pairs as 0-indexed
[i, j] with i < j) and results/12_test_sets.json (counts per set and every dropped sequence with
its reason) plus its .meta.json.
"""
import fnmatch
import io
import json
import tarfile
import zipfile
from collections import Counter
from pathlib import PurePosixPath

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.structures import PARSERS
from rnalm_exposure.test_sets import filter_sets, normalize


def archive_members(path, inner=None):
    """(name, bytes) of every file in a .tar.gz or .zip; with inner, of the zip nested inside it."""
    if inner:
        with zipfile.ZipFile(path) as z:
            path = io.BytesIO(z.read(inner))
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith("/"):
                    yield name, z.read(name)
    else:
        with tarfile.open(path) as t:
            for m in t:
                if m.isfile():
                    yield m.name, t.extractfile(m).read()


def read_set(name, cfg, raw_dir):
    parse = PARSERS[cfg["format"]]
    records = []
    for member, data in archive_members(raw_dir / cfg["archive"], cfg.get("inner_archive")):
        if not fnmatch.fnmatchcase(member, cfg["members"]):
            continue
        seq, pairs = parse(data.decode())
        stem = PurePosixPath(member).stem
        records.append({"seq_id": f"{name}:{stem}", "dataset": name, "file": member,
                        "sequence": normalize(seq), "pairs": pairs})
    if not records:
        raise FileNotFoundError(f"{name}: no members match {cfg['members']}")
    if len({r["seq_id"] for r in records}) != len(records):
        raise ValueError(f"{name}: file stems are not unique")
    return records


def config_value(config, dotted):
    """Follow a reference such as models.rna_fm.max_input_length to its value."""
    value = config
    for key in dotted.split("."):
        value = value[key]
    return value


def main():
    config = load_config()
    paths, cfg = config["paths"], config["test_sets"]
    raw_dir = repo_path(paths["raw_dir"])
    max_length = config_value(config, cfg["length_cutoff"])
    priority = cfg["duplicates"]["across_sets"]
    records = {name: read_set(name, cfg["sets"][name], raw_dir) for name in priority}

    kept, dropped = filter_sets(records, max_length, priority)
    # Sequences dropped from one set for conflicting copies but kept in a later set: the across-set
    # rule only sees copies that survived their own set's checks.
    conflicted = {r["sequence"] for r, reason, _ in dropped if reason == "duplicate_within_set_conflicting"}
    kept_after_conflict = sorted(r["seq_id"] for r in kept if r["sequence"] in conflicted)

    out_path = repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in kept:
            f.write(json.dumps({**r, "pairs": sorted(r["pairs"])}) + "\n")

    summary = {}
    for name in priority:
        reasons = Counter(reason for r, reason, _ in dropped if r["dataset"] == name)
        summary[name] = {"files": len(records[name]), **reasons,
                         "kept": sum(r["dataset"] == name for r in kept)}
        print(name, json.dumps(summary[name]), flush=True)
    result = {
        "max_length": max_length, "priority": priority, "output": display_path(out_path),
        "kept": len(kept), "per_set": summary, "kept_after_conflict_elsewhere": kept_after_conflict,
        "dropped": [{"seq_id": r["seq_id"], "reason": reason, "detail": detail} for r, reason, detail in dropped],
    }
    print(f"{len(kept)} test sequences; kept after a conflict in another set: {kept_after_conflict}")
    out = repo_path(paths["results_dir"]) / "12_test_sets.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    write_meta(out, config)


if __name__ == "__main__":
    main()
