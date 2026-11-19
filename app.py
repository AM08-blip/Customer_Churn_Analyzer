from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

app = FastAPI(
    title="Customer Churn Analyzer",
    description="A machine-learning application for predicting customer churn.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

model_bundle = None


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=1000)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.on_event("startup")
def load_trained_model():
    global model_bundle
    if MODEL_PATH.exists():
        model_bundle = joblib.load(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": model_bundle is not None
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run: python -m src.model_training"
        )

    try:
        customer_df = pd.DataFrame([customer.model_dump()])
        pipeline = model_bundle["pipeline"]
        model_name = model_bundle["model_name"]

        probability = float(pipeline.predict_proba(customer_df)[0][1])
        prediction = int(probability >= 0.50)

        if probability >= 0.70:
            risk = "High"
        elif probability >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk,
            "model_used": model_name,
            "decision_threshold": 50
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
