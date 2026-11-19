# Customer Churn Analyzer

A college-level customer churn prediction project built with FastAPI, scikit-learn, XGBoost, HTML and CSS.

## 1. Dataset

Download the IBM Telco Customer Churn dataset from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Extract `WA_Fn-UseC_-Telco-Customer-Churn.csv`, place it in `data/`, and rename it to `customer_churn.csv`.

## 2. Setup (Windows PowerShell)

```powershell
py -3.11 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train and evaluate

Run from the main project folder:

```powershell
python -m src.model_training
python -m src.model_evaluation
```

Training compares Decision Tree, Random Forest and XGBoost using a held-out test set and saves the model with the highest test ROC-AUC to `models/churn_model.joblib`.

## 4. Run the web application

```powershell
uvicorn app:app --reload
```

Open:
- Main application: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Train the model before opening the prediction page. The model file is created by the training command.
