#!/usr/bin/env bash
# Reproduces all results of the article. Place the two input files in data/ first (see README.md).
set -e
cd "$(dirname "$0")/scripts"
python run_rq12.py GBM
python run_rq12.py MLR,SAM
python summarize_rq12.py
python ratio_consistent.py
python run_grouped_trees.py
python tree_selection.py
python epa_external.py
python robustness.py
python run_rq3.py
python summarize_rq3.py
python run_nested_loco.py
python summarize_nested_loco.py
python epa_eval.py
python epa_safety_factors.py
python fig1_lines.py
python fig1_plot.py
python figures.py
