def risk_band(probability):
    if probability >= 0.60:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"

def review_guidance(probability):
    band = risk_band(probability)
    if band == "High":
        return "Refer for enhanced human review; verify information and affordability."
    if band == "Medium":
        return "Review affordability and supporting documentation before a decision."
    return "Continue standard review; this score is not an approval decision."
