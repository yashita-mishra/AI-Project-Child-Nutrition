# Child Nutrition Predictor

Causal Bayesian network on NHANES 2021–2023 (children 2–6): ethnicity, parent education, income,
WIC, SNAP, food security → anaemia. Missing data handled with EM; do() simulator; held-out tests; dashboard.

## Run
    pip install -r requirements.txt
    python -m unittest discover -s tests -v     # 19 tests
    python -m src.run_all                       # ~2 min: tuning, masking, CV, scenarios -> results/, figures/
    python -m src.build_ui                      # -> app/index.html (open in a browser; works offline)

## Layout
- `src/bbn.py` – the network: CPTs (Dirichlet prior), EM, variable elimination, do(), backdoor, d-separation
- `src/data.py` – node definitions, DAG, loader for `data/children_model.csv`
- `src/experiments.py` – masking experiments, cross-validation, scenarios with bootstrap intervals
- `src/run_all.py`, `src/build_ui.py` – one-command reproduction
- `results/` – CSVs behind every number; `figures/` – plots; `app/` – dashboard (`template.html` + fonts)
- `tests/test_bbn.py` – unit tests (EM monotone, VE = enumeration, do() = backdoor, d-separation, ...)

## Known limits
- 340 labelled children, 24 anaemic: intervals are wide; AUC 0.64 (logistic regression 0.71).
- WIC/SNAP are markers of need as well as treatments; do(WIC) / do(SNAP) are not clean causal effects.
- Data reject the missing parent_edu -> WIC/SNAP/food_security arrows (results/implied_independence_checks.csv).
- do() drops observed evidence on the intervened node and its descendants (approximation, not a full counterfactual).
