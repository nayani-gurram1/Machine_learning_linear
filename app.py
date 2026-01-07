import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report

# --------------------------------------------------
# PAGE CONFIG (Dark Dashboard)
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM DARK THEME CSS
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.kpi-box {
    background-color: #161b22;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.kpi-title {
    font-size: 14px;
    color: #9da5b4;
}
.kpi-value {
    font-size: 26px;
    font-weight: bold;
}
.section {
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("## 📊 Telco Customer Churn Dashboard")
st.caption("Compact, clean dashboard to analyze customer churn and predict churn probability.")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
total = len(df)
leaving = (df["Churn"] == "Yes").sum()
staying = (df["Churn"] == "No").sum()
churn_rate = leaving / total * 100

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Total Customers</div>
    <div class="kpi-value">{total}</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Leaving</div>
    <div class="kpi-value" style="color:#ff4b4b;">{leaving}</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Staying</div>
    <div class="kpi-value" style="color:#2ecc71;">{staying}</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Churn Rate</div>
    <div class="kpi-value">{churn_rate:.2f}%</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SAMPLE DATA
# --------------------------------------------------
st.markdown("###  Sample Customer Data")
st.dataframe(df.head(8), use_container_width=True)

# --------------------------------------------------
# CHURN DISTRIBUTION
# --------------------------------------------------
st.markdown("###  Customer Distribution")

fig, ax = plt.subplots()
df["Churn"].value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Customers")
st.pyplot(fig)

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
X = df.drop(["Churn", "customerID"], axis=1, errors="ignore")
y = df["Churn"].map({"Yes": 1, "No": 0})

# Convert TotalCharges to numeric
if "TotalCharges" in X.columns:
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

# Fill missing values
X = X.fillna(0)
# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Save feature names for prediction
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Stay", "Leave"], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.markdown("###  Model Performance")
st.metric("Accuracy", f"{accuracy*100:.2f}%")
st.markdown("### Classification Report")
st.dataframe(report_df.style.format({
    'precision': "{:.2f}",
    'recall': "{:.2f}",
    'f1-score': "{:.2f}",
    'support': "{:.0f}"
}), use_container_width=True)

# --------------------------------------------------
# CONFUSION MATRIX + ROC
# --------------------------------------------------
st.markdown("###  Model Evaluation")
col1, col2 = st.columns(2)

# Confusion Matrix
with col1:
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["Stay", "Leave"])
    ax.set_yticklabels(["Stay", "Leave"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

    st.pyplot(fig)
y_prob = model.predict_proba(X_test)[:, 1]

# ROC Curve
with col2:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# --------------------------------------------------
# PREDICT CHURN (COMPACT)
# --------------------------------------------------
st.markdown("###  Predict Customer Churn")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0, 2000.0)

sample = X.mean().to_frame().T
sample["tenure"] = tenure
sample["MonthlyCharges"] = monthly
sample["TotalCharges"] = total_charges

sample_scaled = scaler.transform(sample)

if st.button("Predict Churn"):
    sample = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )   

# Assign numeric inputs
    sample["tenure"] = tenure
    sample["MonthlyCharges"] = monthly
    sample["TotalCharges"] = total_charges

    sample_scaled = scaler.transform(sample)

    pred = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0][1]
    pred = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0][1]

    if pred == 1:
        st.error(f" Customer Likely to Churn (Probability: {prob:.2f})")
    else:
        st.success(f" Customer Likely to Stay (Probability: {prob:.2f})")
