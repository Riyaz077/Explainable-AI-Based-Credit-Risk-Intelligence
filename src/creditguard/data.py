import pandas as pd
REQUIRED = ["application_id","annual_income","loan_amount","loan_term_months",
"interest_rate","credit_score","debt_to_income","employment_years",
"delinquencies","credit_utilization","loan_purpose","defaulted"]

def load_data(path):
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.application_id.duplicated().any():
        raise ValueError("application_id must be unique")
    return df
