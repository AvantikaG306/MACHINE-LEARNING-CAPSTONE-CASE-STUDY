# Machine Learning Capstone Case Study (23CSE301)

An end-to-end Machine Learning case study project exploring dual tracks: **Classification** (Online Shoppers Purchasing Intention) and **Regression** (Ames Housing Price Prediction). This project covers full pipelines from exploratory data analysis, data cleaning, feature engineering, baseline & advanced modeling, hyperparameter tuning, cross-validation, to model selection and visualization.

---

## Project Overview

This repository contains two main Jupyter Notebook workflows located in the `notebook/` directory:

1. **`notebook/classification.ipynb`**: Predicts whether an online shopping session will result in a purchase (`Revenue` = True/False) using **5 classification algorithms** (Part-A Review 1).
2. **`notebook/regression.ipynb`**: Predicts continuous residential property prices (`SalePrice`) using **10 regression algorithms**, hyperparameter tuning via `GridSearchCV`, and 5-fold cross-validation.

Both tracks enforce a fixed seed (`random_state=42`) across all random splits and estimators for complete reproducibility.

---

## Track 1: Classification Track (`notebook/classification.ipynb`)

### Objective
Predict session-level purchase intent (**`Revenue`**) on the Online Shoppers Purchasing Intention Dataset.

* **Dataset**: `data/online_shoppers_intention.csv` (12,330 sessions, 18 features: 10 numerical, 8 categorical)
* **Target Variable**: `Revenue` (Binary: `True` / `False`, Class Imbalance: 84.5% False / 15.5% True)

### Workflow & Methodology
1. **Audit & Data Cleaning**: Audited column types, checked for missing values, and confirmed baseline distributions.
2. **Exploratory Data Analysis (EDA)**: Analyzed revenue conversion rates against key features (`PageValues`, `BounceRates`, `ExitRates`, `Month`, `VisitorType`).
3. **Preprocessing & Feature Engineering**:
   - Engineered interaction features: `TotalDuration` and `ProductEngagementRatio`.
   - Encoded categorical variables using One-Hot Encoding and Label Encoding.
   - Applied `StandardScaler` within pipelines for scaling-sensitive models (`Logistic Regression`, `KNN`, `SVC`).
   - Employed `StratifiedKFold` train-test splitting (80/20 ratio) to preserve class balance.
4. **Model Benchmarking (5 Classifiers)**:
   - Decision Tree (`max_depth=5`)
   - Support Vector Machine (`SVC`, $C=10$, `kernel='rbf'`)
   - Logistic Regression ($C=1.0$)
   - K-Nearest Neighbors (`KNN`, $k=5$, Euclidean distance)
   - Gaussian Naive Bayes (`GaussianNB`)

### Model Performance Comparison

| Rank | Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | TN | FP | FN | TP |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Decision Tree ($depth=5$)** | **0.8959** | **0.8942** | **0.8959** | **0.8950** | **0.9197** | 1940 | 119 | 135 | 247 |
| **2** | **SVC ($C=10, \text{rbf}$)** | **0.8923** | **0.8836** | **0.8923** | **0.8848** | **0.8600** | 1981 | 78 | 185 | 197 |
| **3** | **Logistic Regression** | 0.8861 | 0.8764 | 0.8861 | 0.8698 | 0.8984 | 2015 | 44 | 234 | 148 |
| **4** | **KNN ($k=5$)** | 0.8775 | 0.8641 | 0.8775 | 0.8648 | 0.7893 | 1984 | 75 | 224 | 158 |
| **5** | **Gaussian Naive Bayes** | 0.8460 | 0.8534 | 0.8460 | 0.8493 | 0.8383 | 1849 | 210 | 166 | 216 |

### 5-Fold Stratified Cross-Validation (Top-2 Models)

| Model | Mean CV Accuracy | CV Accuracy Std | Mean CV Weighted F1 | CV Weighted F1 Std | Mean CV ROC-AUC | CV ROC-AUC Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree ($depth=5$)** | **0.8953** | $\pm 0.0064$ | **0.8903** | $\pm 0.0081$ | **0.9158** | $\pm 0.0071$ |
| **SVC ($C=10, \text{rbf}$)** | **0.8911** | $\pm 0.0034$ | **0.8833** | $\pm 0.0041$ | **0.8577** | $\pm 0.0097$ |

### Key Insights
- `PageValues` emerged as the single most critical feature for predicting shopper conversion.
- Decision Tree ($depth=5$) achieved the top Weighted F1 (0.8950) and highest ROC-AUC (0.9197).
- Low cross-validation variance ($\sigma < 0.009$) across folds confirms model stability and strong generalization capability.

---

## Track 2: Regression Track (`notebook/regression.ipynb`)

### Objective
Predict residential house sales prices (`SalePrice`) in Ames, Iowa.

* **Dataset**: `data/housing.csv` (2,930 records, 82 features)
* **Target Variable**: `SalePrice` (Continuous monetary target in USD)

### Workflow & Methodology
1. **Data Preprocessing & Feature Engineering**:
   - Engineered composite structural metrics: `TotalSF` (`1stFlrSF` + `2ndFlrSF` + `TotalBsmtSF`), `HouseAge` (`YrSold` - `YearBuilt`), and `TotalBath`.
   - Handled skewed numerical features and missing values.
   - Encoded ordinal attributes and converted nominal variables using One-Hot Encoding.
2. **Model Benchmarking (10 Regressors)**: Evaluated linear models, regularized regressions, distance-based regressors, decision trees, ensemble models, and kernel SVR.
3. **Hyperparameter Tuning**: Performed exhaustive grid search (`GridSearchCV`) for top models.
4. **Validation & Diagnostics**: 5-Fold Cross-Validation, residual plot diagnostics, predicted vs. actual analysis, and feature importance extraction.

### Model Performance Comparison (Test Set Benchmark)

| Rank | Model | $R^2$ | RMSE ($) | MAE ($) |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Gradient Boosting ($lr=0.05$)** | **0.9238** | **19,554.26** | **12,639.86** |
| **2** | **Random Forest ($n=200$)** | **0.9125** | **20,959.23** | **13,514.19** |
| **3** | **Lasso Regression ($\alpha=100$)** | **0.9020** | **22,176.11** | **15,951.50** |
| **4** | **Ridge Regression ($\alpha=10$)** | 0.8991 | 22,503.13 | 16,120.21 |
| **5** | **Linear Regression** | 0.8982 | 22,596.58 | 16,177.65 |
| **6** | **KNN Regressor ($k=7$)** | 0.8742 | 25,122.55 | 17,679.55 |
| **7** | **Decision Tree ($depth=6$)** | 0.8351 | 28,766.48 | 20,131.02 |
| **8** | **Polynomial Regression ($degree=2$)** | 0.7932 | 32,209.09 | 17,791.71 |
| **9** | **ElasticNet ($\alpha=100, l1=0.5$)** | 0.3157 | 58,596.14 | 43,408.01 |
| **10** | **SVR ($C=10, \text{rbf}$)** | -0.0441 | 72,379.94 | 51,271.39 |

### Hyperparameter Tuning Results (`GridSearchCV`)

| Model | Baseline $R^2$ | Tuned $R^2$ | $R^2$ Improvement |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | 0.9174 | **0.9196** | $+0.0022$ |
| **Random Forest Regressor** | 0.9124 | **0.9175** | $+0.0051$ |

### 5-Fold Cross-Validation (Tuned Models)

| Model | 5-Fold CV $R^2$ Scores | Mean CV $R^2$ | CV $R^2$ Std |
| :--- | :--- | :---: | :---: |
| **Gradient Boosting Tuned** | `[0.9418, 0.9147, 0.9371, 0.9224, 0.9237]` | **0.9279** | $\pm 0.0100$ |
| **Random Forest Tuned** | `[0.9279, 0.8918, 0.9167, 0.9015, 0.9189]` | **0.9113** | $\pm 0.0130$ |

### Key Insights
- Gradient Boosting Regressor emerged as the **best overall model** ($R^2 = 0.9238$, Mean CV $R^2 = 0.9279$).
- Ensembles (Gradient Boosting, Random Forest) significantly outperformed standalone linear models and distance/kernel models.
- Key price drivers: `OverallQual`, `GrLivArea`, `TotalSF`, `2ndFlrSF`, and `Neighborhood`.

---

## Repository Structure

```text
MACHINE-LEARNING-CAPSTONE-CASE-STUDY/
├── data/
│   ├── housing.csv                     # Dataset for regression track
│   └── online_shoppers_intention.csv   # Dataset for classification track
├── notebook/
│   ├── classification.ipynb            # Online Shoppers Purchasing Intention notebook
│   └── regression.ipynb                # Ames Housing price prediction notebook
├── requirements.txt                    # Required Python packages and versions
└── README.md                           # Project overview, benchmark results & setup guide
```

---

## Setup & Installation Instructions

### Prerequisites
- Python 3.9 or higher installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/amudhannnnn/MACHINE-LEARNING-CAPSTONE-CASE-STUDY.git
cd MACHINE-LEARNING-CAPSTONE-CASE-STUDY
```

### 2. Create and Activate Virtual Environment
* **On macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **On Windows**:
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```
Or open JupyterLab:
```bash
jupyter lab
```

Navigate to the `notebook/` directory to run `classification.ipynb` or `regression.ipynb`.

---

## Reproducibility Notice

All experiments utilize a fixed random seed (`random_state = 42`) across data splits, pipeline transformations, and model initializations. Re-running `notebook/classification.ipynb` or `notebook/regression.ipynb` from top to bottom will yield identical numerical metrics, figures, and model evaluations.
