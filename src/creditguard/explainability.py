def reason_codes(application):
    """Simple transparent review flags; not legally sufficient adverse-action reasons."""
    reasons = []
    if application["debt_to_income"] >= 0.40:
        reasons.append("High debt-to-income ratio")
    if application["credit_score"] < 620:
        reasons.append("Low credit score")
    if application["delinquencies"] >= 2:
        reasons.append("Multiple reported delinquencies")
    if application["credit_utilization"] >= 0.75:
        reasons.append("High revolving credit utilization")
    if application["loan_amount"] / max(application["annual_income"], 1) >= 0.50:
        reasons.append("High requested loan relative to annual income")
    return reasons or ["No major rule-based risk flags identified"]
