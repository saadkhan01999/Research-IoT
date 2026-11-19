# Data

## Source

CICIoT2023, Canadian Institute for Cybersecurity, University of New Brunswick.
Official page: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>

> E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu and A. A. Ghorbani,
> "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment,"
> *Sensors*, vol. 23, no. 13, p. 5941, 2023. <https://doi.org/10.3390/s23135941>

The dataset is not stored in this repository: it is 1.4 GB compressed, and the official page
does not state terms for re-publishing it. Please download it from the page above.

## Release used

The per-capture CSV release: 309 CSV files, one per packet capture, in 34 folders (one per
label). Each file has the same 39 feature columns and no label column; the label is the folder
name. This is not the 46-feature CSV release that many other studies use.

## How the single file was built

`scripts/prep/merge_per_pcap_csvs.py` reads every CSV inside the downloaded `CSV.zip`, adds two
columns and writes one file, `CICIoT2023_merged.zip` (one CSV inside, 9.95 GB):

- `Label`: the folder name (34 labels; the folder `Benign_Final` becomes `BenignTraffic`);
- `Category`: the attack family of the dataset paper (Benign, DDoS, DoS, Mirai, Recon,
  Spoofing, Web, BruteForce).

Rows keep their original values and order. Nothing is shuffled or removed.
`scripts/prep/verify_merge.py` checks the result against the source files.

- `raw/merge_report.txt`: class counts and the row count of every source file, in the order
  they appear in the merged file. Because of this order, the source capture of every row is
  known, which the file-grouped and ordered splits need.
- `checksums/raw_sha256.csv`: size and SHA-256 of the merged zip and of each file inside it.
  `python run.py verify-data` compares a local copy with these values.

The merged file goes to `data/raw/CICIoT2023_merged.zip`. `data/interim/` and
`data/processed/` are caches written by the scripts.

## Sample files

`sample/` holds three files of 5,000 rows each (0.03% of the data), with the 39 features and
the 34-class label in the last column. They were drawn from the test rows of the experiments
of the deduplicated random split (the capture stream from the ordered split) by
`scripts/13_export_model.py`:

| File | Rows | What it is |
|---|---|---|
| `natural_mix.csv` | 5,000 | test rows drawn with the class mix of the data |
| `balanced.csv` | 5,000 | 625 test rows from each of the 8 classes |
| `capture_stream.csv` | 5,000 | 25 blocks of 200 consecutive rows, each block from a different capture; two extra columns give the source file and the row number inside it |

They are meant for a first look at the data and for trying the trained model. Results in the
paper are computed on the full dataset, not on these files.
