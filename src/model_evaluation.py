from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from src.data_preprocessing import load_dataset, prepare_data


MODEL_PATH = Path("models/churn_model.joblib")


def main():
    if not MODEL_PATH.exists():
        print("Trained model not found. Run: python -m src.model_training")
        return

    df = load_dataset()
    X_train, X_test, y_train, y_test, _ = prepare_data(df)

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("MODEL:", bundle["model_name"])
    print("\nEVALUATION RESULTS")
    print("------------------")
    print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, predictions, zero_division=0), 4))
    print("F1-score :", round(f1_score(y_test, predictions, zero_division=0), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, probabilities), 4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
