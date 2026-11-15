# Pretraining exposure in RNA language models

Does the zero-shot secondary structure accuracy of RNA language models depend on how well each
test sequence and its evolutionary relatives are represented in the models' pretraining corpora?

For every sequence in standard RNA structure benchmarks, this project counts its homologs in a
reconstruction of each model's pretraining corpus, scores the model's zero-shot structure
prediction (one attention head, symmetrized, APC-corrected, thresholded into base pairs, F1), and
fits a mixed-effects regression of F1 on exposure with Rfam family as the grouping variable. A
continued-training experiment then injects homologs of chosen test families into training, and
masked-letter and verbatim-span recovery tests compare sequences the models saw in pretraining
with matched sequences they did not.

Models: RNA-FM; RiNALMo micro, mega and giga; AIDO.RNA (GB.RNA) 650M and 1.6B; AIDO.DNA 300M
as a DNA language model floor. RNA-MSM appears only in the reproduction of Wang et al. (2026),
since its single-sequence F1 is near zero.
Test sets: ArchiveII, bpRNA-TS0, bpRNA-new, Archive-NoFam.

## Status

Work in progress (data collection Nov 2026 – Feb 2027).

- [x] Pretraining corpora reconstructed: RNA-FM (RNAcentral 19.0, chosen among five archived releases by deduplicated size and length distribution), RiNALMo (approximated), AIDO.RNA, next-release and post-snapshot corpora
- [x] Homolog search (MMseqs2), Rfam family assignment (Infernal) and ViennaRNA baseline
- [x] Reproduction of Wang et al. (2026) on their selection and test structures
- [x] Attention heads chosen on low-exposure validation sequences and frozen (`results/frozen_heads.json`)
- [x] Zero-shot scoring of all test sets, mixed-effects models and checks (permutation, next release, APC off, placebo, cluster and same-family counts, nonlinear fits)
- [x] Masked-letter recovery on seen vs unseen sequences
- [ ] Continued-training experiment with injected homologs (RiNALMo micro done; mega running)
- [ ] Verbatim span recovery
- [ ] Paper

## Installation

```bash
conda env create -f environment.yml
conda activate rnalm-exposure
pytest
```

This environment covers downloads, sequence searches, ViennaRNA and the statistics (R with lme4
and lmerTest for script 19). `environment-extra.yml` is a separate environment for the shape
checks, figures and recovery analyses. Model inference runs on a CUDA GPU with PyTorch, in two
environments because their `transformers` versions conflict: `environment-infer.yml` (RNA-FM and
RiNALMo through the `multimolecule` Hugging Face ports) and `environment-aido.yml` (AIDO models
through GenBio's `modelgenerator`).

## Pipeline

Scripts in `scripts/` are numbered in run order. Every setting lives in
`configs/config.yaml`, and every script writes one output plus a `*.meta.json` recording the git
commit, the full config and package versions.

| Scripts | Step |
|---|---|
| `00`–`06` | Download archived RNAcentral releases, count them, deduplicate at 100% identity (exact containment on both strands, the rule of cd-hit-est -c 1.0 -r 1, checked against cd-hit-est on subsets) and compare each candidate with RNA-FM's published corpus statistics |
| `07`, `10`, `11` | Download test and validation sets, Rfam, RNA3DB and RefSeq; build the RiNALMo and AIDO.RNA corpora |
| `08`, `09` | GPU memory check per model; reproduction of Wang et al. (2026) |
| `12`–`16` | Prepare test sets; homolog counts (≥ 80% identity over ≥ 80% of the query); ViennaRNA baseline; Rfam families; exact-match, max-identity checks |
| `17`, `18` | Choose and freeze each model's attention head; score all test sequences |
| `19`–`25` | Mixed-effects models and checks (`19_fit_models.R`), placebo corpus, cluster and same-family counts, shape checks, figures, head specialization by exposure |
| `26`–`29` | Continued-training experiment: injection sets, training, checkpoint scoring, analysis |
| `30`–`33` | Masked-letter recovery on matched seen and unseen sequences, overall and by cluster size |
| `34`–`36` | Verbatim span recovery (spans of 5–50 nt) on the same matched sequences |

## Repository layout

```
configs/              config.yaml: every path, threshold and setting
data_manifests/       what was downloaded from where (URL, size, MD5)
scripts/              numbered pipeline steps
src/rnalm_exposure/   shared code (corpora and dedup, homolog search, readout, metrics, structures, verbatim test)
tests/                pytest
results/              result tables, figures and their *.meta.json
```

`data/` is not tracked; the download scripts rebuild it from the manifests.

## Data

RNAcentral archived releases: https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/

## Citation

Paper in preparation.

## License

MIT
