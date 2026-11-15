from pathlib import Path
from scripts.generate_data import generate
from src.creditguard.pipeline import run

if __name__ == "__main__":
    p=Path("data/sample/loan_applications.csv")
    if not p.exists():
        p.parent.mkdir(parents=True,exist_ok=True)
        generate().to_csv(p,index=False)
    name, metrics, _=run(str(p))
    print("Selected model:",name)
    print(metrics)
