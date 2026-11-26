# Transferability of machine learning predictions of GAC breakthrough to new source waters

Analysis code and derived results for the article

> D. Altan Koç, *Transferability of machine learning predictions of granular activated carbon breakthrough to new source waters* (submitted).

The study re-examines the published gradient boosting model for the treatable volume to 10 % breakthrough
(BV10) of granular activated carbon adsorbers (Koyama *et al.*, *Environ. Sci. Technol.*, 2024, **58**, 17114)
under validation schemes that withhold entire source studies (leave-one-study-out, LOSO) or entire source
waters (leave-one-campaign-out, LOCO), derives safety factors for a tolerated probability of premature
breakthrough, and tests the models on independent EPA pilot data.

## Input data (not redistributed here)

Both input data sets are publicly available. Download them and place them in `data/`:

| File | Source |
|---|---|
| `data/Breakthrough Database.xlsx` | Koyama *et al.* (2024), https://github.com/furtman/GAC_BV10_estimation (file `data/Breakthrough Database.xlsx`), licence CC BY-SA 4.0 |
| `data/Phos_VOC_SciHubDataset.xlsx` | Haupert *et al.* (2023), *AWWA Water Sci.* **5**, e1361; EPA ScienceHub, https://doi.org/10.23719/1529105 |

## Installation

Python 3.13 was used. Install the dependencies with

```
pip install -r requirements.txt
```

## Running the analysis

All scripts are in `scripts/` and write to `results/`. Run them from inside `scripts/`, in this order
(or run `bash run_all.sh` from the repository root):

| Step | Script | Output in `results/` | Used in the article |
|---|---|---|---|
| 1 | `run_rq12.py GBM`, `run_rq12.py MLR,SAM`, `run_rq12.py MEAN`, `run_rq12.py SAM_NOINT` | `predictions_*.csv` | out-of-sample predictions for all schemes (MEAN: null predictor; SAM_NOINT: SAM without interactions) |
| 2 | `summarize_rq12.py` | `table_rq12_metrics.csv`, `table_loso_per_study.csv`, `table_rq2_paired.csv` | Tables 2, 3, 5 and 6 |
| 3 | `ratio_consistent.py`, `stats_paired.py` | `robust_ratio_studyweighted.csv`, `robust_ratio_paired.csv`, `robust_rho_diff.csv`, `robust_tost.csv`, `robust_summaries.csv` | Table 3, Section 3.5 (paired bootstrap, TOST, median errors, weighting) |
| 4 | `run_grouped_trees.py`, then `tree_selection.py` | `robust_predictions_GBM_groupedtrees.csv`, `robust_tree_selection.csv` | Table 4 |
| 4b | `run_gbm_leaves.py`, then `summarize_gbm_leaves.py` | `robust_predictions_GBM_leaves.csv`, `robust_gbm_leaves.csv` | sensitivity to J, Section 3.3 |
| 4c | `run_logac.py`, then `summarize_logac.py` | `robust_predictions_LOGAC.csv`, `robust_logac_summary.csv` | leave-one-adsorbent-out, Section 3.3 |
| 5 | `epa_external.py` | `epa_external.csv` | Table 8 |
| 6 | `robustness.py` | `robust_bootstrap_ci.csv`, `robust_safety_factor.csv`, `robust_pilot_overprediction_ci.csv`, `robust_flag_sensitivity.csv`, `robust_epa_domain_*.csv` | Table 3 (CIs), Table 7, Sections 3.3, 3.4 and 3.7 |
| 7 | `run_rq3.py`, then `summarize_rq3.py` | `rq3_bounds.csv`, `table_rq3_tradeoff.csv` | nested study-level calibration, Section 3.6 |
| 8 | `run_nested_loco.py`, then `summarize_nested_loco.py` | `nested_loco_bounds.csv`, `robust_nested_loco.csv` | nested campaign-level calibration, Section 3.6 |
| 8b | `controlling_compound.py` | `robust_controlling_compound.csv` | controlling compound, Section 3.6 |
| 9 | `epa_eval.py`, `epa_safety_factors.py`, `epa_sensitivity.py` | `table_epa_summary.csv`, `robust_epa_loco_factor.csv`, `robust_epa_sensitivity.csv` | Table 9, Section 3.7 |
| 10 | `fig1_lines.py`, `fig1_plot.py`, `figures.py` | `fig1_validation_schemes.png/.pdf`, `fig2_loso_per_study.png` | Figs. 1 and 2 |

Steps 1, 4, 4b, 4c, 7 and 8 refit the models many times and are the time-consuming part of the analysis;
the other steps take seconds to a few minutes. Random number generators are seeded, so all results
are reproducible.

`common.py` contains data loading, the construction of the 12 source-water campaigns from the
18 source studies, the three model classes (GBM, SAM, MLR) and the performance metrics.
`paths.py` defines all file locations relative to the repository root.

## Derived results

The `results/` folder contains the outputs of the scripts as used in the article, so that the
tables and figures can be checked without rerunning the models.

## Software versions

scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, joblib 1.6.0, openpyxl 3.1.5.

## Licence

The code is released under the MIT licence (see `LICENSE`). The files in `results/` are derived from the
Koyama *et al.* database and are therefore shared under the same CC BY-SA 4.0 licence.

## Citation

If you use this code, please cite the article above and the original database by Koyama *et al.* (2024).
