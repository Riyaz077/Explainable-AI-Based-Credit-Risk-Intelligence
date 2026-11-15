import pandas as pd

NUMERIC = ["annual_income","loan_amount","loan_term_months","interest_rate",
"credit_score","debt_to_income","employment_years","delinquencies","credit_utilization"]

def make_features(df):
    x = df[NUMERIC].copy()
    x["loan_to_income"] = x.loan_amount / x.annual_income.clip(lower=1)
    x["estimated_monthly_payment_proxy"] = (
        x.loan_amount * (1 + x.interest_rate / 100 * x.loan_term_months / 12)
        / x.loan_term_months.clip(lower=1)
    )
    x["income_after_payment_ratio"] = (
        x.annual_income / 12 / x.estimated_monthly_payment_proxy.clip(lower=1)
    )
    x["purpose_business"] = (df.loan_purpose.astype(str).str.lower() == "business").astype(int)
    x["purpose_auto"] = (df.loan_purpose.astype(str).str.lower() == "auto").astype(int)
    x["purpose_education"] = (df.loan_purpose.astype(str).str.lower() == "education").astype(int)
    return x.replace([float("inf"), -float("inf")], 0).fillna(0)
