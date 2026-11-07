# Data manifests

One TSV per external source, listing exactly what was downloaded. `bytes` is the size
reported by the server; `md5` is computed on the downloaded file (EBI does not publish
per-file checksums for these releases). `scripts/00_download_rnacentral.py` downloads every
row and checks both.

| File | Source |
|---|---|
| `rnacentral.tsv` | RNAcentral archived releases (https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/): `rnacentral_active.fasta.gz` per release, plus the inactive files of 15.0 and 16.0 and the species-specific file of 15.0. Releases 15.0-19.0: RNA-FM corpus and next-release candidates; 22.0: RiNALMo corpus; 24.0 and 28.0: post-snapshot corpus and its exclusions |
