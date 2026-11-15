from pathlib import Path
import json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import load_data
from .features import make_features
from .models import build_logistic, build_forest, evaluate

def run(data_path="data/sample/loan_applications.csv", output_dir="artifacts"):
    df = load_data(data_path)
    X, y = make_features(df), df.defaulted.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {"logistic_regression": build_logistic(), "random_forest": build_forest()}
    results, fitted = {}, {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        results[name] = evaluate(y_test, prob)
        fitted[name] = model
    best_name = max(results, key=lambda k: results[k]["roc_auc"] or 0)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], out / "credit_model.joblib")
    (out / "metrics.json").write_text(json.dumps({"selected_model": best_name, "models": results}, indent=2))
    scored = df.copy()
    scored["default_probability"] = fitted[best_name].predict_proba(X)[:, 1]
    scored.to_csv("data/processed/scored_applications.csv", index=False)
    return best_name, results, scored
