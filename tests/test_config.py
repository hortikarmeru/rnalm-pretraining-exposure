from rnalm_exposure.config import load_config, repo_path


def test_config_loads():
    config = load_config()
    assert "paths" in config and "corpora" in config


def test_manifest_matches_candidate_releases():
    config = load_config()
    with open(repo_path(config["paths"]["manifests"]["rnacentral"])) as f:
        listed = {line.split("\t")[0] for line in f.read().splitlines()[1:]}
    assert set(config["corpora"]["rna_fm"]["candidate_releases"]) <= listed
