# Interconnect Customer Churn Prediction

## Description
Final project of the TripleTen Data Science bootcamp. A churn prediction system is built for **Interconnect**, a telecommunications operator, to rank active customers by their probability of cancelling and trigger targeted retention campaigns.

- **Data cutoff:** 2020-02-01
- **Coverage:** 7,043 unique customers (5,174 active + 1,869 churned)
- **Goal:** AUC ROC >= 0.75 (minimum acceptable) / >= 0.88 (stretch)

## Dataset
Source: Internal Interconnect CSV exports (not versioned by data policy — must be placed manually in `Data/`).

| File | Content |
|------|---------|
| `contract.csv` | Contract type, billing, charges, dates |
| `personal.csv` | Demographics (gender, senior, partner, dependents) |
| `internet.csv` | Internet service and add-ons |
| `phone.csv` | Phone service and multiple lines |

- **Records:** 7,043 customers after merge
- **Class balance:** 73 % active / 27 % churned (handled with oversampling on train only)
- **Target:** `Churn` derived from `EndDate` vs cutoff

## Results
| # | Model | AUC Test | AUC CV (5-fold) |
|---|-------|---------:|----------------:|
| 0 | DummyClassifier (Baseline) | 0.500 | — |
| 1 | DecisionTree | 0.805 | 0.827 ± 0.011 |
| 2 | MLP (sklearn) | 0.798 | 0.840 ± 0.009 |
| 3 | XGBoost base | 0.828 | 0.842 ± 0.008 |
| 4 | CatBoost | 0.836 | 0.855 ± 0.008 |
| 5 | LightGBM | 0.842 | 0.850 ± 0.006 |
| 6 | RandomForest | 0.845 | 0.852 ± 0.007 |
| 7 | LogisticRegression | 0.846 | 0.849 ± 0.010 |
| 8 | XGBoost tuned | **0.848** | 0.857 |
| 9 | **Ensemble Top 3 (XGB + CatBoost + RF)** | **0.847** | **0.859 ± 0.007** |

**Best model:** Ensemble Top 3 — AUC 0.847 test / 0.859 CV. Chosen over the single best (XGBoost tuned, 0.848 test) for its higher CV score and greater robustness. Goal achieved (>= 0.75); the gap to 0.88 requires additional data sources (consumption, support tickets) not available in this phase.

## Pipeline
1. **Load & quality audit** — per-dataset null/dup checks across the four CSV files.
2. **Consolidated null audit (post-merge)** — every NaN traced to a probable cause; no real missing data.
3. **EDA** — tenure, contract type, year-over-year churn evolution.
4. **LTV calculation + statistical validation** (Mann-Whitney) on churned vs active.
5. **Feature engineering** — `AddonsCount`, `FiberNoSecurity`, `Tenure` (replaces leaky `DaysActive`).
6. **Seven base models** with synced best hyperparameters + 5-fold stratified CV.
7. **Tuning** — `GridSearchCV` (DT / LogReg / RF) and `RandomizedSearchCV` (LGBM / XGBoost / CatBoost) with `verbose=2` and early stopping on boosters.
8. **MLP** as `MLPClassifier` inside a `Pipeline` with `StandardScaler` + native `early_stopping=True`.
9. **Ensembling** — average of top 3 + LogReg stacking.
10. **Comparison table** — AUC with and without CV to confirm stability.
11. **Feature importance consensus** across 6 models.
12. **Marketing export** — CSV/XLSX with 5,174 active customers in 3 risk segments and a recommended action per segment.

## Tech Stack
| Category | Tools |
|----------|-------|
| ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| Tuning | GridSearchCV, RandomizedSearchCV, early stopping |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Deep Learning | PyTorch (CUDA, optional for MLP experiments) |

## Hardware Used
- **CPU:** 32 cores (leveraged with `n_jobs=-1`)
- **RAM:** 64 GB
- **GPU:** NVIDIA RTX 4080 SUPER (optional, used for PyTorch experiments)

## Structure
```
Proyecto Final/
├── README.md
├── requirements.txt
├── Data/
│   ├── contract.csv
│   ├── personal.csv
│   ├── internet.csv
│   └── phone.csv
└── Notebook/
    ├── Proyecto Final.ipynb            # Main notebook
    ├── Informe Mesa Directiva.ipynb    # Executive summary
    ├── Plan de Trabajo.ipynb           # Work plan & retrospective
    ├── ERD_editor.html                 # Entity-relationship diagram
    ├── lista_retencion.csv             # Generated at runtime
    └── lista_retencion.xlsx            # Generated at runtime
```

## How to Run
1. Activate the shared virtual environment:
   ```
   ..\..\.venv\Scripts\activate       # Windows
   source ../../.venv/bin/activate    # macOS / Linux
   ```
2. Install dependencies (first time only):
   ```
   pip install -r requirements.txt
   ```
3. Optional — install PyTorch with CUDA for GPU on Windows:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
4. Place the four CSV files in `Data/`.
5. Open and run the main notebook:
   ```
   jupyter notebook "Notebook/Proyecto Final.ipynb"
   ```
   (Kernel → Restart & Run All. Total runtime: ~3-4 min on GPU, ~7-10 min on CPU.)

`lista_retencion.csv` and `lista_retencion.xlsx` are generated automatically with the 5,174 active customers segmented by risk and recommended action.

## Conclusions
- **Goal achieved:** 8 of 9 models surpass the AUC >= 0.75 threshold; the realistic ceiling with the available features is ~0.86.
- **Best model:** Ensemble Top 3 (XGBoost tuned + CatBoost + RandomForest), AUC 0.847 test / 0.859 CV — selected for stability over the single best (XGBoost tuned, 0.848 test).
- **Stability:** CV vs test gap is <= 0.04 across all models — no severe overfitting.
- **Business finding:** Accumulated churn loss ≈ **2.7 M USD** (15 % of total LTV). Two-thirds of the leak comes from Month-to-month customers (churn rate 28 % in 2014 → 55 % in 2019). Annual and two-year contracts hold steady at 13 % and 4 %.
- **Top drivers (consensus of 6 models):** contract type (two-year contract slows churn), tenure, monthly charges, fiber optic service (accelerates churn), value-added services (slow churn).
- **Immediate recommendation:** migrate Month-to-month customers to annual or two-year contracts. The prioritized marketing list is generated automatically.
- **For low-latency production:** Logistic Regression (AUC 0.846) is the best practical choice — near-instant prediction, easy to maintain, almost identical AUC to the ensemble.
- **For maximum accuracy:** Ensemble Top 3 offers the most robust ranking of customers by churn risk.

## Author
Fernando Muerza — TripleTen Data Science, Final Project (Sprint 17).
