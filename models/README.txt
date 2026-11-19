CUSTOMER CHURN PIPELINE UPGRADE

Replace these two files in your existing project:
  src/data_preprocessing.py
  src/model_training.py

This version adds:
- TenureGroup, AverageMonthlySpend, ServicesCount
- Logistic Regression, Decision Tree, Random Forest, XGBoost
- RandomizedSearchCV with stratified 5-fold cross-validation
- Cross-validation PR-AUC model selection
- Test accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix
- Saved model bundle compatible with the existing API's bundle['pipeline'] key

IMPORTANT:
Feature engineering is included inside the saved scikit-learn pipeline. This is
important because the API can continue sending the original customer fields;
the trained pipeline creates the engineered fields automatically.

Run from the project root (with your venv activated):
  python -m src.data_preprocessing
  python -m src.model_training

The tuning process may take several minutes depending on your computer.
