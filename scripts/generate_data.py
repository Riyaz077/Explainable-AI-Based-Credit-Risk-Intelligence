from pathlib import Path
import numpy as np
import pandas as pd
def generate(n=3000, seed=42):
    r=np.random.default_rng(seed)
    income=np.clip(r.lognormal(10.8,.55,n),12000,250000)
    amount=np.clip(income*r.uniform(.05,.65,n),1000,100000)
    term=r.choice([12,24,36,48,60],n)
    rate=np.clip(r.normal(11.5,3,n),4,25)
    score=np.clip(r.normal(690,65,n),400,850).astype(int)
    dti=np.clip(r.beta(2.2,5,n)*.85,.01,.9)
    emp=np.clip(r.gamma(2.2,3,n),0,35)
    delin=np.clip(r.poisson(.55,n),0,8)
    util=np.clip(r.beta(2,3,n),0,1)
    purpose=r.choice(['personal','auto','education','business'],n,p=[.45,.25,.15,.15])
    z=-3.2+3.4*dti+.018*(650-score)+.55*delin+1.25*util+.9*(amount/income)+.025*(rate-10)
    p=1/(1+np.exp(-z)); y=r.binomial(1,p)
    return pd.DataFrame({'application_id':[f'LG{i:06d}' for i in range(1,n+1)],
    'annual_income':income.round(2),'loan_amount':amount.round(2),'loan_term_months':term,
    'interest_rate':rate.round(2),'credit_score':score,'debt_to_income':dti.round(3),
    'employment_years':emp.round(1),'delinquencies':delin,'credit_utilization':util.round(3),
    'loan_purpose':purpose,'defaulted':y})
if __name__=='__main__':
    p=Path('data/sample/loan_applications.csv'); p.parent.mkdir(parents=True,exist_ok=True)
    generate().to_csv(p,index=False)
