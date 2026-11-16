# CreditGuard
## Explainable AI-Based Credit Risk and Loan Default Prediction System

CreditGuard is an end-to-end machine learning demonstrator for assessing loan default risk, estimating default probabilities, explaining model outputs, and supporting consistent credit-review workflows. It includes a reproducible training pipeline, synthetic loan dataset, Streamlit dashboard, FastAPI prediction endpoint, and model evaluation utilities.

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

## **Quick Start**

- Set up and activate a Python virtual environment.
- Install all project dependencies using `requirements.txt`.
- Generate the sample dataset using the data generation script.
- Execute the end-to-end pipeline for data preprocessing and model training.
- Launch the interactive Streamlit dashboard for analytics and visualization.
- Start the FastAPI backend to access prediction services.
- Explore available API endpoints through the interactive Swagger documentation.
- Run automated tests using `pytest` to verify project functionality.

## **API Usage – Loan Risk Prediction**

The `/predict` API endpoint accepts a POST request containing financial, credit, and employment-related information to evaluate loan risk and generate predictive insights.

**Key input parameters:**

- **Financial Details:** Annual income, loan amount, loan term, and interest rate.
- **Credit Profile:** Credit score, debt-to-income ratio, credit utilization, and previous delinquencies.
- **Employment Details:** Employment duration in years.
- **Loan Information:** Loan purpose and borrowing requirements.

**Expected output:**

- Predicted loan risk classification.
- Estimated probability of default.
- Creditworthiness assessment.
- Risk-based lending insights and recommendations.

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
