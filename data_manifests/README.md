# Data manifests

One TSV per external source, listing exactly what was downloaded. `bytes` is the size
reported by the server; `md5` is computed on the downloaded file (EBI does not publish
per-file checksums for these releases). `scripts/00_download_rnacentral.py` downloads every
row and checks both.

| File | Source |
|---|---|
| `rnacentral.tsv` | RNAcentral archived releases, `rnacentral_active.fasta.gz` (https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/) |
