# CreditGuard
## Explainable AI-Based Credit Risk and Loan Default Prediction System

CreditGuard is an end-to-end machine learning demonstrator for assessing loan default risk, estimating default probabilities, explaining model outputs, and supporting consistent credit-review workflows. It includes a reproducible training pipeline, synthetic loan dataset, Streamlit dashboard, FastAPI prediction endpoint, and model evaluation utilities.

> **Important:** The included dataset is synthetic and created for demonstration. This project is not a production credit-scoring system and must not be used to make real lending decisions without extensive validation, legal review, fairness testing, and human oversight.

## Features
- Loan default probability prediction using supervised machine learning.
- Risk-band assignment: Low, Medium, and High.
- Explainability through feature contribution summaries and human-readable reason codes.
- Data validation and preprocessing pipeline.
- Model evaluation using ROC-AUC, precision, recall, F1, confusion matrix, and calibration-oriented review.
- Streamlit dashboard for portfolio-level and individual application exploration.
- FastAPI endpoint for single-application scoring.
- Synthetic dataset generator and automated tests.

## Architecture
```text
Synthetic Loan Applications
          |
          v
 Data Validation & Cleaning
          |
          v
 Feature Engineering
          |
          v
 Logistic Regression / Random Forest
          |
          v
 Probability + Risk Band
          |
          v
 Explainability & Reason Codes
          |
       +--+--+
       |     |
       v     v
  Dashboard  FastAPI
```

## Technology
Python · Pandas · NumPy · Scikit-learn · SHAP-ready design · Streamlit · FastAPI · Joblib · Pytest

## Repository structure
```text
CreditGuard/
├── api/main.py
├── config/config.yaml
├── dashboard/app.py
├── data/sample/loan_applications.csv
├── data/processed/.gitkeep
├── docs/                 # project and technical documentation
├── notebooks/            # EDA, modelling, explainability
├── scripts/
│   ├── generate_data.py
│   └── run_pipeline.py
├── src/creditguard/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── explainability.py
│   ├── policy.py
│   └── pipeline.py
├── tests/
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

## Quick start
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python scripts/generate_data.py
python scripts/run_pipeline.py
```

Launch the dashboard:
```bash
streamlit run dashboard/app.py
```

Launch the API:
```bash
uvicorn api.main:app --reload
```
API docs: `http://127.0.0.1:8000/docs`

Run tests:
```bash
pytest -q
```

## Example API request
`POST /predict`
```json
{
  "annual_income": 65000,
  "loan_amount": 18000,
  "loan_term_months": 36,
  "interest_rate": 10.5,
  "credit_score": 710,
  "debt_to_income": 0.28,
  "employment_years": 5,
  "delinquencies": 0,
  "credit_utilization": 0.32,
  "loan_purpose": "personal"
}
```

## Methodology
1. Generate or load loan application records.
2. Validate required fields and permissible ranges.
3. Engineer affordability and credit-behaviour features.
4. Split data into training and holdout sets.
5. Train Logistic Regression and Random Forest classifiers.
6. Evaluate probability ranking and classification performance.
7. Produce feature-based reason codes for review.
8. Score applications and display portfolio insights.

## Explainability
The baseline explainability module provides transparent, rule-based reason codes based on an applicant's input features (for example, high debt-to-income ratio or recent delinquencies). These are indicative model-review aids, not legally sufficient adverse-action reasons. The project does not claim to implement SHAP unless separately added and validated.

## Responsible use
Credit decisions can materially affect people's lives. Do not use protected attributes or proxies to discriminate. Test for disparate impact and model stability, document data provenance, provide meaningful human review and appeal processes, protect personal information, and comply with applicable lending and data-protection laws. Do not commit real applicant data, secrets, or credentials to GitHub.

## Limitations and future work
The current version uses synthetic data, a simplified target-generation process, baseline models, and heuristic reason codes. Future extensions may include independently validated real-world datasets, calibration, temporal validation, fairness metrics, model monitoring, SHAP explanations, secure authentication, and governed lending-system integration.

## Author

**Mohamed Riyaz S.**  
Integrated M.Tech, Computer Science and Engineering (Business Analytics)  
VIT Chennai
