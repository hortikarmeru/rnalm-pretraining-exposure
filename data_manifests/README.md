# Data manifests

One TSV per external source, listing exactly what was downloaded. `bytes` is the size
reported by the server; `md5` is computed on the downloaded file (EBI does not publish
per-file checksums for these releases). `scripts/00_download_rnacentral.py` downloads every
row and checks both.

| File | Source |
|---|---|
| `rnacentral.tsv` | RNAcentral archived releases (https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/): `rnacentral_active.fasta.gz` per release, plus the inactive and species-specific files of 15.0 |
