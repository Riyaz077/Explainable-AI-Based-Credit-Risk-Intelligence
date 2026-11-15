import pandas as pd
from src.creditguard.features import make_features

def test_feature_engineering():
    row={"annual_income":60000,"loan_amount":12000,"loan_term_months":24,
         "interest_rate":10,"credit_score":700,"debt_to_income":0.25,
         "employment_years":4,"delinquencies":0,"credit_utilization":0.3,
         "loan_purpose":"personal"}
    x=make_features(pd.DataFrame([row]))
    assert "loan_to_income" in x.columns
    assert x.iloc[0]["loan_to_income"] == 0.2
