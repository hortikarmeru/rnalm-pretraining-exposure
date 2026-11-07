# Pretraining exposure in RNA language models

Does the zero-shot secondary structure accuracy of RNA language models depend on how well each
test sequence and its evolutionary relatives are represented in the models' pretraining corpora?

For every sequence in standard RNA structure benchmarks, this project counts its homologs in a
reconstruction of each model's pretraining corpus, scores the model's zero-shot structure
prediction (one attention head, symmetrized, APC-corrected, thresholded into base pairs, F1), and
fits a mixed-effects regression of F1 on exposure with structural family as the grouping variable.

Models: RNA-FM, RiNALMo (micro, mega, giga), RNA-MSM (single-sequence), and a DNA language model.
Test sets: ArchiveII, bpRNA-TS0, bpRNA-new, Archive-NoFam.

## Status

Work in progress (data collection Nov 2026 – Feb 2027).

- [x] RNAcentral candidate releases for the RNA-FM corpus downloaded, counted and deduplicated at 100% identity
- [ ] RNA-FM corpus release chosen
- [ ] RiNALMo corpus (RNAcentral 22 + Rfam 14.9 + RefSeq 216 ncRNA) and placebo corpus
- [ ] Homolog search (MMseqs2) and family assignment (Infernal)
- [ ] Attention head selection on low-exposure validation sequences (`results/frozen_heads.json`)
- [ ] Zero-shot scoring and regression

## Installation

```bash
conda env create -f environment.yml
conda activate rnalm-exposure
pytest
```

This environment covers downloads, sequence searches, ViennaRNA and the statistics. Model
inference runs on a CUDA GPU and needs PyTorch and the model packages; see the notes at the end
of `environment.yml`.

## Pipeline

Scripts in `scripts/` are numbered in run order. Every setting lives in
`configs/config.yaml`, and every script writes one output plus a `*.meta.json` recording the git
commit, the full config and package versions.

| Script | Output |
|---|---|
| `00_download_rnacentral.py` | RNAcentral releases in `data/raw/rnacentral/<release>/`, checked against `data_manifests/rnacentral.tsv` |
| `01_count_rnacentral_releases.py` | `results/01_rnacentral_release_counts.json`: records and distinct sequences per candidate release |
| `02_dedup_cdhit.py` | cd-hit-est at 100% on a candidate (weeks on a full release; kept as the reference that script 04 checks against) |
| `03_estimate_dedup_size.py` | `results/03_dedup_estimates.json`: cluster count per candidate estimated from a random sample |
| `04_check_dedup_vs_cdhit.py` | `results/04_dedup_check.json`: containment dedup vs cd-hit-est on subsets (16 of 300,000 records differ) |
| `05_dedup_containment.py` | `data/processed/rnacentral_cdhit100/<name>.fasta` and `results/05_dedup_<name>.json`: 100% dedup of a candidate by exact containment, the rule of cd-hit-est -c 1.0 -r 1 |
| `06_corpus_fingerprints.py` | `results/06_corpus_fingerprints.json`: length distribution and rare letters of each deduplicated candidate vs RNA-FM's published corpus statistics |
| `07_download_references.py` | test and validation sets, Rfam, RNA3DB and the RefSeq catalog in `data/raw/`, checked against `data_manifests/downloads.tsv` |
| `08_check_model_memory.py` | `results/08_model_memory.json`: peak GPU memory with full attention maps at the input limit, per model |

## Repository layout

```
configs/          config.yaml: every path, threshold and setting
data_manifests/   what was downloaded from where (URL, size, MD5)
scripts/          numbered pipeline steps
src/rnalm_exposure/   shared code (config loading, FASTA handling, exact 100% dedup, output metadata)
tests/            pytest
results/          small result tables and their *.meta.json
```

`data/` is not tracked; `scripts/00_download_rnacentral.py` rebuilds it from the manifests.

## Data

RNAcentral archived releases: https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/

## Citation

Paper in preparation.

## License

MIT
