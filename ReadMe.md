# Optimizing Post-Discharge Interventions for Diabetes Patients

A **Predict-Then-Optimize (PTO)** framework that helps hospitals allocate limited intervention budgets to reduce 30-day readmissions among diabetic patients.


## Overview

Hospitals operating under value-based care models face financial penalties for excessive readmissions (via Medicare's [HRRP](https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp)). With constrained budgets, the key question is: **which patients should receive post-discharge interventions to maximize cost savings?**

This project answers that question through a two-stage pipeline:

1. **Predict** — estimate each patient's readmission risk using machine learning
2. **Optimize** — allocate interventions using a knapsack-style model to maximize expected savings within a fixed budget


## Results at a Glance

| Scenario | Budget | Encounters Treated | Net Savings | ROI |
|---|---|---|---|---|
| Baseline | $1M | 1,832 | ~$2.24M | 2.24 |
| Doubled budget | $2M | 3,228 | ~$3.88M | 1.94 |
| Reduced effectiveness (−5%) | $1M | 1,832 | ~$1.70M | 1.70 |

> **Best model:** HistGradientBoosting — ROC AUC 0.697, Log Loss 0.624, Average Precision 0.657


## Project Structure

```
Project/
├── diabetic_data.csv       # Encounter-level data (UCI ML Repository, 130 U.S. hospitals, 1999–2008)
├── IDs_mapping.csv         # Mappings for admission/discharge/source ID fields
├── requirements.txt        # Python dependencies
└── PTO_diabetes.ipynb      # End-to-end notebook: preprocessing → modeling → optimization
```


## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Open the notebook**
```bash
jupyter notebook PTO_diabetes.ipynb
```

The notebook walks through all steps in order: data preprocessing, model training and evaluation, optimization, and what-if analysis.


## Methodology

### Prediction Stage

Four classifiers were trained and compared using 5-fold stratified group cross-validation (grouped by patient to prevent data leakage):

| Model | ROC AUC | Log Loss | Avg Precision |
|---|---|---|---|
| Logistic Regression | 0.681 | 0.644 | 0.629 |
| Random Forest | 0.689 | 0.631 | 0.641 |
| XGBoost | 0.692 | 0.628 | 0.649 |
| **HistGradientBoosting** | **0.697** | **0.624** | **0.657** |

HistGradientBoosting was selected for its best-in-class calibration and discriminative power.

### Optimization Stage

The optimization is formulated as a **0-1 knapsack problem**:

$$\max \sum_{i=1}^{n} X_i \left( e \cdot P_i \cdot C_{r,a(i)} - C_{\text{int},a(i)} \right)$$

$$\text{subject to: } \sum_{i=1}^{n} X_i \cdot C_{\text{int},a(i)} \leq B, \quad X_i \in \{0, 1\}$$

| Symbol | Description |
|---|---|
| $X_i$ | Binary: 1 if patient $i$ receives an intervention |
| $P_i$ | Predicted readmission probability for encounter $i$ |
| $e$ | Intervention effectiveness (reduction in readmission probability) |
| $C_{r,a(i)}$ | Readmission cost for patient's age group |
| $C_{\text{int},a(i)}$ | Intervention cost for patient's age group |
| $B$ | Total budget |

Each patient contributes an expected net savings term. The model selects the subset that maximizes total savings without exceeding the budget.


## Data

- **Source:** [UCI ML Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Size:** ~100,000 encounter records across 130 U.S. hospitals
- **Target variable:** Binary readmission (0 = not readmitted, 1 = readmitted within or after 30 days)
- **Key preprocessing steps:**
  - Categorical IDs mapped using `IDs_mapping.csv`
  - Columns with ≥40% missing values dropped; remaining missingness preserved as a `"Missing"` category
  - Group-based train/test split (by patient) to prevent leakage


## Limitations

- **Missing clinical variables:** A1C, glucose serum, and weight were too sparse to include, which may over-index on administrative features.
- **No longitudinal modeling:** Data is encounter-level; repeated visits per patient exist but patient trajectories are not tracked over time.
- **Estimated costs:** Readmission and intervention costs are approximated from published trends and assumed fixed within age groups — not sourced from operational data.
- **No fairness constraints:** The optimization maximizes expected cost savings only. Equitable allocation across demographic groups is not enforced.

*For academic and research purposes only.*