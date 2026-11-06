import csv

from rnalm_exposure.config import load_config, repo_path


def test_config_loads():
    config = load_config()
    assert "paths" in config and "corpora" in config


def test_every_candidate_file_is_in_the_manifest():
    config = load_config()
    with open(repo_path(config["paths"]["manifests"]["rnacentral"])) as f:
        listed = {(r["release"], r["file"]) for r in csv.DictReader(f, delimiter="\t")}
    for c in config["corpora"]["rna_fm"]["candidates"]:
        for file in c["files"]:
            assert (c["release"], file) in listed
