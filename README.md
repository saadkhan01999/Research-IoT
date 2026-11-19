# Machine Learning-Based Multi-Class Attacks Prediction for IoT Devices

A leakage-audited evaluation of IoT attack classifiers on the CICIoT2023 dataset.

MPhil research, Department of Computer Science, Abdul Wali Khan University Mardan.
Author: Saad bin Saeed Khan. The paper is in preparation.

Many studies report multi-class accuracies near 99% on the CICIoT2023 IoT attack dataset,
but most do not say how they split the data. This repository holds the code, the results and
a description of the data for a study that checks those numbers. The data was audited before
any model was trained, and the same models were then trained and tested under four ways of
splitting the data, so that only the split changes between runs.

Every number in the results comes from a script in this repository. Nothing in `results/`
or `paper/figures/` was edited by hand.

## Main findings

**The data limits what any model can reach.** Of the 46,776,697 rows, 55.0% are exact copies
of another row, and 30.6% have a feature vector that also appears under a different label
(mostly a DDoS flood and the DoS flood of the same protocol). No classifier that sees only
these features can pass 92.89% accuracy on the 8 classes or 89.85% on the 34 labels
(`results/labels/ceilings.json`).

**The usual setup does not give 99%.** With a random split and library defaults, the best
model reaches 85.79% accuracy on the 8 classes (XGBoost).

**Copies are not the main problem; captures are.** Best macro-F1 on the 8 classes, mean of
three seeds (`results/analysis/main_results.csv`, `paper/figures/fig07_main_results.png`):

| Split | Best macro-F1 | Best accuracy | Highest possible accuracy on these test rows |
|---|---|---|---|
| Random | 0.673 (LightGBM) | 83.89% | 93.68% |
| Random, copies removed | 0.677 (LightGBM) | 83.32% | 97.87% |
| Whole capture files held out | 0.649 (RF) | 79.68% | 99.76% |
| Earlier rows of each capture train, later rows test | 0.676 (RF) | 82.55% | 99.86% |

Removing copies changes macro-F1 by at most 0.011. Holding out whole capture files lowers it
by 4.7% on the 8 classes and 6.4% on the 34 labels, and the result then depends on which
captures are held out: the spread over seeds (0.031) is as large as the gap between the best
and the worst model (0.030).

**Other results**

- *Unseen attack types:* when one attack type is removed from training, its rows are still
  called an attack, but often put in the wrong family. 23 of 32 types lose more than 5 points;
  all four DoS floods are then called DDoS. At most 22.2% of the rows of a held-out type are
  called benign (`results/experiments/unseen_type__*`, `paper/figures/fig13_unseen_type.png`).
- *Class imbalance:* class weights, undersampling and SMOTE-Tomek raise the recall of rare
  classes in all 12 combinations tested, and lower precision in all 12. With XGBoost on the
  8 classes, SMOTE-Tomek raises recall on the rare families from 15.2% to 30.8% while their
  precision falls from 77.9% to 6.0%.
- *Model choice:* LightGBM ranks first, but over all protocol and label-scheme blocks it is not
  significantly better than XGBoost (Holm-adjusted p = 0.057), and XGBoost labels rows in
  batches 18 times faster on the test laptop (7,649 against 423 rows per second, one thread).
- *Neural models:* a small MLP stays below XGBoost on single rows (0.634 against 0.676). On
  windows of 10 consecutive rows, XGBoost on simple window statistics does best (0.793).
- *Early detection:* one row gives XGBoost the right family for 86.7% of captures; a vote over
  50 rows gives 90.4%.
- *Metrics:* capping and tuning raised the best macro-F1 from 0.663 to 0.673, but the share of
  benign rows that XGBoost flags as an attack rose from 12.8% to 42.8%. Accuracy or macro-F1
  alone would hide this.

## Dataset

CICIoT2023 was created by the Canadian Institute for Cybersecurity, University of New
Brunswick. Download it from the official page:
<https://www.unb.ca/cic/datasets/iotdataset-2023.html>

> E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu and A. A. Ghorbani,
> "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment,"
> *Sensors*, vol. 23, no. 13, p. 5941, 2023. <https://doi.org/10.3390/s23135941>

The data itself is not in this repository. It is 1.4 GB compressed (9.9 GB as one CSV), and
the official page does not state terms for re-publishing it. What is here instead:

- `data/README.md`: which release is used and how it was turned into one labelled file.
- `data/checksums/raw_sha256.csv`: SHA-256 of the merged file, so a copy can be checked
  (`python run.py verify-data`).
- `data/raw/merge_report.txt`: the row count of every one of the 309 source files, in order.
- `scripts/prep/`: the scripts that merged the 309 per-capture CSV files into one file.
- `data/sample/`: three files of 5,000 labelled rows each, for a first look at the data.

## Repository layout

```
config/          experiment settings: profiles (default, quick, full), models, label schemes
data/            checksums, merge report, sample files and data/README.md
src/iotml/       the Python package: data loading, audit, splits, sampling, models, metrics, statistics
scripts/         one numbered script per step (00 to 13); scripts/prep built the merged raw file
results/         outputs of the full runs (CSV and JSON), described below
paper/figures/   figures made by scripts/09_figures.py (PNG and PDF)
paper/tables/    tables made by scripts/09_tables.py (LaTeX)
tests/           unit tests
run.py           one entry point for every step (python run.py --list)
```

### What is in `results/`

| Folder or file | Content |
|---|---|
| `audit/` | Duplicates, conflicting labels, column profile, order inside captures |
| `labels/` | Label schemes, merged-label groups, accuracy ceilings, sampling plan |
| `splits/` | Rows in each part of every split, and the ceiling of each test part |
| `baseline/` | Random-split baseline as other studies run it, and the careful baseline |
| `experiments/` | One JSON file per training run (358 runs): settings, timing, every metric, per-class results, confusion matrix; tuned settings in `random/tuning/` |
| `analysis/` | Mean results, model ranking tests, bootstrap intervals, calibration, SHAP, cost |
| `early_detection/` | Share of captures labelled correctly from the first n rows |
| `dashboard/` | Record of the dashboard screenshots used in the figures |
| `number_trace.csv` | Every reported number with the file it comes from |
| `env/` | Machine and library versions used |

## Reproducing the results

Windows or Linux, Python 3.11:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt   # Linux: .venv/bin/python
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python run.py test
```

Put the merged file at `data/raw/CICIoT2023_merged.zip` (build it from the official release
with `scripts/prep/merge_per_pcap_csvs.py`), then run the steps in order:

```bash
python run.py verify-data      # check the file against data/checksums
python run.py cache            # Parquet cache of the 46.8 million rows
python run.py audit            # duplicates, conflicts, ceilings
python run.py labels           # label schemes and sampling plan
python run.py baseline-full    # random-split baseline
python run.py row-index
python run.py strict           # four split protocols and the unseen-type test
python run.py imbalance
python run.py neural
python run.py analysis         # statistics, calibration, SHAP, cost
python run.py early
python run.py results          # figures, tables and number_trace.csv
```

Every step skips runs whose result file already exists, so an interrupted run continues where
it stopped. Each `*-quick` task and `--profile quick` run the same code on small samples in
minutes, which is useful for checking the setup.

**Run time.** All results were produced on one laptop: Intel Core i5-6300U (2 cores, 4
threads), 8 GB RAM, no GPU, Windows 10. The full set of experiments took about five days, most
of it the four split protocols. Seeds: 11, 23 and 37 for every capped and tuned
configuration, and all five (11, 23, 37, 53, 71) for the as-published baseline; 20 tuning
trials per model.

## Models and settings

Decision tree, random forest, extra trees, XGBoost and LightGBM, tuned once on the random
split and reused unchanged under the other protocols; a small MLP, a 1D-CNN and a GRU for the
neural comparison. Training uses a sample capped per label; test rows are weighted back to the
natural class mix, so the scores describe the traffic as it was recorded. The settings are in
`config/` and the tuned values in `results/experiments/random/tuning/`.
