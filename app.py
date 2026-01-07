import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title(" Customer Churn Prediction System")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

st.subheader(" Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
X = df.drop(["Churn", "customerID"], axis=1, errors="ignore")
y = df["Churn"].map({"Yes": 1, "No": 0})

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Train Model
# -------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------------------------
# Performance Section (HIGHLIGHTED)
# -------------------------------------------------
st.subheader(" Model Performance")

# Accuracy Highlight
st.metric(
    label="Model Accuracy",
    value=f"{accuracy*100:.2f} %"
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No Churn", "Actual Churn"],
    columns=["Predicted No Churn", "Predicted Churn"]
)

st.subheader(" Confusion Matrix")
st.dataframe(
    cm_df.style
        .background_gradient(cmap="Blues")
        .set_properties(**{"font-weight": "bold"})
)

# Confusion Matrix Analysis
tn, fp, fn, tp = cm.ravel()

st.markdown("###  Confusion Matrix Analysis")
st.write(f" Correctly identified **non-churn** customers: {tn}")
st.write(f" Non-churn customers misclassified as churn: {fp}")
st.write(f" Churn customers missed by the model: {fn}")
st.write(f" Correctly identified **churn** customers: {tp}")

# Performance Message
if accuracy >= 0.8:
    st.success(" Model performance is GOOD")
elif accuracy >= 0.6:
    st.warning(" Model performance is AVERAGE")
else:
    st.error(" Model performance is POOR")

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.subheader(" Predict Churn for a New Customer")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f" Customer is **Likely to Churn** (Probability: {probability:.2f})")
    else:
        st.success(f" Customer is **Likely to Stay** (Probability: {probability:.2f})")
