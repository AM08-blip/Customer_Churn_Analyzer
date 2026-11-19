import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


DATA_PATH = "data/customer_churn.csv"

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_dataset(path=DATA_PATH):
    """Load the IBM Telco churn dataset and perform basic data cleaning."""
    df = pd.read_csv(path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain a 'Churn' target column.")

    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1})

    if df["Churn"].isna().any():
        raise ValueError("Unexpected or missing values found in the Churn column.")

    df["Churn"] = df["Churn"].astype(int)
    return df


def engineer_features(X):
    """Create engineered features. This function is also used by the API pipeline."""
    X = X.copy()

    if "tenure" in X.columns:
        tenure = pd.to_numeric(X["tenure"], errors="coerce")
        X["TenureGroup"] = pd.cut(
            tenure,
            bins=[-1, 12, 24, 48, float("inf")],
            labels=["0-12 months", "13-24 months", "25-48 months", "Over 48 months"],
        ).astype("object")

    if {"TotalCharges", "tenure"}.issubset(X.columns):
        total = pd.to_numeric(X["TotalCharges"], errors="coerce")
        tenure = pd.to_numeric(X["tenure"], errors="coerce")
        X["AverageMonthlySpend"] = total.div(tenure.where(tenure > 0))
        if "MonthlyCharges" in X.columns:
            X["AverageMonthlySpend"] = X["AverageMonthlySpend"].fillna(
                pd.to_numeric(X["MonthlyCharges"], errors="coerce")
            )

    available = [column for column in SERVICE_COLUMNS if column in X.columns]
    if available:
        X["ServicesCount"] = sum(
            X[column].astype(str).str.strip().eq("Yes").astype(int)
            for column in available
        )

    return X


def prepare_data(df):
    """Split raw customer columns; feature engineering remains inside the ML pipeline."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Infer the transformed column names from training data only.
    engineered_train = engineer_features(X_train)
    numerical_features = engineered_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = engineered_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    dataset = load_dataset()
    X_train, X_test, y_train, y_test, _ = prepare_data(dataset)
    print("Dataset loaded successfully.")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Original input features:", X_train.shape[1])
    print("Engineered features: TenureGroup, AverageMonthlySpend, ServicesCount")
