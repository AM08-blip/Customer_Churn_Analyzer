from pathlib import Path
import json
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

from src.data_preprocessing import load_dataset, prepare_data, engineer_features


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_model_searches():
    """Return the four required classifiers and their tuning spaces."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0, 50.0],
                "classifier__solver": ["liblinear", "lbfgs"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {
                "classifier__max_depth": [3, 5, 7, 10, None],
                "classifier__min_samples_split": [2, 5, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 5, 10],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
            {
                "classifier__n_estimators": [150, 250, 350],
                "classifier__max_depth": [8, 12, 16, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", 0.8],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1),
            {
                "classifier__n_estimators": [150, 250, 350],
                "classifier__max_depth": [3, 4, 5],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__subsample": [0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.8, 0.9, 1.0],
            },
        ),
    }


def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "decision_threshold": 0.50,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def main():
    print("Loading customer churn dataset...")
    df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_cv_score = -1.0
    best_bundle = None

    for model_name, (classifier, parameter_space) in build_model_searches().items():
        print(f"\\nTuning {model_name} with 5-fold cross-validation...")
        pipeline = Pipeline([
            ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_space,
            n_iter=10,
            scoring="average_precision",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            refit=True,
            verbose=1,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        metrics = evaluate_model(search.best_estimator_, X_test, y_test)
        result = {
            "model_name": model_name,
            "best_cv_pr_auc": float(search.best_score_),
            "best_params": search.best_params_,
            **metrics,
        }
        results.append(result)

        print("Best CV PR-AUC:", round(search.best_score_, 4))
        print("Best parameters:", search.best_params_)
        print("Test Accuracy :", round(metrics["accuracy"], 4))
        print("Test Precision:", round(metrics["precision"], 4))
        print("Test Recall   :", round(metrics["recall"], 4))
        print("Test F1-score :", round(metrics["f1_score"], 4))
        print("Test ROC-AUC  :", round(metrics["roc_auc"], 4))
        print("Test PR-AUC   :", round(metrics["pr_auc"], 4))
        print("Confusion matrix:", metrics["confusion_matrix"])

        # Model selection uses cross-validation PR-AUC, not the held-out test set.
        if search.best_score_ > best_cv_score:
            best_cv_score = float(search.best_score_)
            best_bundle = {
                "pipeline": search.best_estimator_,
                "model_name": model_name,
                "metrics": metrics,
                "best_cv_pr_auc": best_cv_score,
                "best_params": search.best_params_,
                "decision_threshold": 0.50,
            }

    joblib.dump(best_bundle, MODEL_DIR / "churn_model.joblib")
    with (MODEL_DIR / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\\nModel comparison complete.")
    print("Selected model:", best_bundle["model_name"])
    print("Best cross-validation PR-AUC:", round(best_cv_score, 4))
    print("Final test ROC-AUC:", round(best_bundle["metrics"]["roc_auc"], 4))
    print("Final test PR-AUC:", round(best_bundle["metrics"]["pr_auc"], 4))
    print("Saved model: models/churn_model.joblib")
    print("Saved comparison: models/model_comparison.json")

    best_predictions = (
        best_bundle["pipeline"].predict_proba(X_test)[:, 1] >= 0.50
    ).astype(int)
    print("\\nClassification report:")
    print(classification_report(y_test, best_predictions, zero_division=0))


if __name__ == "__main__":
    main()
