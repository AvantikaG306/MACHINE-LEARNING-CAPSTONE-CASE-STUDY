# Avantika G - 306 | Review 2 Individual Contribution

## Role: Classification Track Lead & Ensemble Learning Specialist
**Course:** 23CSE301 Machine Learning | **Review:** 2

---

### Individual Deliverables & Notebooks:
1. `08_classification_part_b_ensembles.ipynb`:
   - Built and trained 5 advanced ensemble algorithms: Random Forest, AdaBoost, Gradient Boosting, Bagging Classifier, and MLP Neural Network.
   - Evaluated models on the held-out stratified test set ($N=2,466$) across Accuracy, Precision, Recall, Weighted F1, and ROC-AUC.
2. `09_classification_tuning_and_cross_validation.ipynb`:
   - Conducted hyperparameter tuning with `GridSearchCV` on the best models (Gradient Boosting and Random Forest).
   - Achieved peak test performance with Tuned Gradient Boosting: Accuracy = **90.74%**, Weighted F1 = **0.9046**, ROC-AUC = **0.9356**.
   - Verified generalization stability using 5-Fold Stratified Cross-Validation ($	ext{Mean CV F1} = 0.8989 \pm 0.0056$).
3. `10_consolidated_10_algorithm_benchmark.ipynb`:
   - Constructed the consolidated 10-algorithm comparison table (Part A baselines + Part B ensembles).
   - Produced multi-model ROC curves, F1 trade-off plots, and confusion matrix diagnostics.

---

### Key Viva Questions & Answers Prepared:
* **Why did Gradient Boosting outperform other classifiers?**  
  Gradient Boosting sequentially corrects residual errors of prior shallow trees using gradient descent on the log-loss function, resulting in superior decision boundary refinement.
* **How was class imbalance addressed?**  
  Through stratified train-test splitting (80:20 preserving 84.5% non-revenue / 15.5% revenue ratio) and evaluating primarily on Weighted F1 and ROC-AUC rather than raw accuracy.
