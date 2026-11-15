from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.creditguard.features import make_features
from src.creditguard.explainability import reason_codes
from src.creditguard.policy import risk_band, review_guidance

app=FastAPI(title="CreditGuard API",version="1.0.0")

class LoanApplication(BaseModel):
    annual_income: float=Field(gt=0)
    loan_amount: float=Field(gt=0)
    loan_term_months: int=Field(ge=1,le=360)
    interest_rate: float=Field(ge=0,le=100)
    credit_score: int=Field(ge=300,le=850)
    debt_to_income: float=Field(ge=0,le=1)
    employment_years: float=Field(ge=0)
    delinquencies: int=Field(ge=0)
    credit_utilization: float=Field(ge=0,le=1)
    loan_purpose: str="personal"

@app.get("/")
def home():
    return {"name":"CreditGuard API","docs":"/docs"}

@app.post("/predict")
def predict(applicant: LoanApplication):
    model_path=Path("artifacts/credit_model.joblib")
    if not model_path.exists():
        raise HTTPException(status_code=503,detail="Train the model first using python scripts/run_pipeline.py")
    data=applicant.model_dump()
    model=joblib.load(model_path)
    X=make_features(pd.DataFrame([data]))
    probability=float(model.predict_proba(X)[0,1])
    return {"default_probability":round(probability,4),
            "risk_band":risk_band(probability),
            "review_guidance":review_guidance(probability),
            "reason_codes":reason_codes(data),
            "notice":"Decision-support demonstration only; not a lending decision."}
