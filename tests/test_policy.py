from src.creditguard.policy import risk_band
from src.creditguard.explainability import reason_codes

def test_risk_bands():
    assert risk_band(0.1) == "Low"
    assert risk_band(0.4) == "Medium"
    assert risk_band(0.8) == "High"

def test_reason_codes_flag_high_dti():
    item={"debt_to_income":0.6,"credit_score":700,"delinquencies":0,
          "credit_utilization":0.2,"loan_amount":10000,"annual_income":50000}
    assert "High debt-to-income ratio" in reason_codes(item)
