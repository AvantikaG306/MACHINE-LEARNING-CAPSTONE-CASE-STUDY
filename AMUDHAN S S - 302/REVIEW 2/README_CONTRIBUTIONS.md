# AMUDHAN S S - 302 | Review 2 Individual Contribution

## Role: Pipeline Integration, Interactive GUI Architect & Cloud Engineering
**Course:** 23CSE301 Machine Learning | **Review:** 2

---

### Individual Deliverables & Components:
1. `08_regression_review2_tuning_and_cv.ipynb`:
   - Full 10-algorithm regression benchmark on Ames Housing dataset ($N=2,930$, 80 features).
   - Hyperparameter tuning on Gradient Boosting Regressor (Rank 1: $R^2 = 0.9238$, $	ext{RMSE} = \$19,554$) and Random Forest (Rank 2: $R^2 = 0.9125$).
   - 5-Fold Cross-Validation on top-2 models ($	ext{GBM Mean CV } R^2 = 0.9279 \pm 0.0100$).
2. `09_classification_review2_master.ipynb` & `10_clustering_review2_master.ipynb`:
   - Full pipeline review and validation for cross-track synthesis.
3. `GUI_Application/` (Interactive Web Platform):
   - **`app.py`**: Streamlit & Plotly enterprise web app featuring 4 views:
     - Executive Dashboard with live KPI benchmark tiles.
     - Ames Property Valuation Engine with 95% valuation confidence intervals.
     - Real-time Purchase Intent Intelligence with conversion probability gauges.
     - Customer Cohort Explorer with 2D interactive PCA mapping and radar profiles.
   - Modernized UI to **100% emoji-free**, clean SVG vector logos, and custom SaaS dark mode styling.
   - **`export_models.py`** & **`models/`**: Serialized inference pipelines (`joblib`) under 1 MB for instantaneous cold-start loading.
4. **Git Infrastructure & Cloud Deployment:**
   - Multi-milestone Git commit architecture with 9 progressive milestones.
   - Deployed live on **Streamlit Community Cloud** (Bonus +2 Marks).

---

### Key Viva Questions & Answers Prepared:
* **How does inference work in the deployed GUI without data leakage?**  
  All preprocessors (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`) are encapsulated inside Scikit-Learn `Pipeline` objects fitted strictly on training data during offline training, then persisted via `joblib`.
* **How was the property valuation confidence interval derived?**  
  Using the model's test-set Root Mean Squared Error (RMSE = $19,554.26): $	ext{Price} \pm 1.96 	imes 	ext{RMSE}$, representing a 95% parametric confidence interval.
