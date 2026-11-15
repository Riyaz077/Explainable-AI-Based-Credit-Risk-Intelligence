from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="CreditGuard", page_icon="🛡️", layout="wide")
st.title("CreditGuard")
st.caption("Explainable AI-Based Credit Risk and Loan Default Prediction")
p=Path("data/processed/scored_applications.csv")
if not p.exists():
    st.warning("Run `python scripts/run_pipeline.py` to train the model and create scored data.")
    st.stop()
df=pd.read_csv(p)
a,b,c=st.columns(3)
a.metric("Applications",f"{len(df):,}")
b.metric("Observed synthetic defaults",f"{df.defaulted.mean():.1%}")
c.metric("Mean predicted default probability",f"{df.default_probability.mean():.1%}")
st.subheader("Predicted default probability")
fig,ax=plt.subplots(); ax.hist(df.default_probability,bins=30); ax.set_xlabel("Probability"); ax.set_ylabel("Applications")
st.pyplot(fig); plt.close(fig)
st.subheader("Risk profile")
df["risk_band"]=pd.cut(df.default_probability,[-.001,.30,.60,1.0],labels=["Low","Medium","High"])
st.bar_chart(df.risk_band.value_counts())
st.subheader("Application explorer")
chosen=st.selectbox("Application ID",df.application_id)
row=df[df.application_id==chosen].iloc[0]
st.write(row.to_frame("Value"))
st.caption("Synthetic demonstration only. Risk scores are not lending decisions.")
