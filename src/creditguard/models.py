from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

def build_logistic(random_state=42):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1500, class_weight="balanced",
                                                  random_state=random_state))])

def build_forest(random_state=42):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                         class_weight="balanced", random_state=random_state, n_jobs=-1))])

def evaluate(y, probability, threshold=0.5):
    pred = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(set(y)) > 1 else None
    }
