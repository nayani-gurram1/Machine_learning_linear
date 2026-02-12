import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="NYC Taxi Hotspot Detection", layout="centered")

st.title("🚕 NYC Taxi Pickup Hotspot Detection using DBSCAN")

# ==============================
# Load Dataset (Fixed Path)
# ==============================

dataset = pd.read_csv(r"C:\Users\gurra\HIERARCHICALCLUSTERING\train.csv")


# Sample to avoid MemoryError
dataset = dataset.sample(10000, random_state=42)

st.subheader("Dataset Preview")
st.dataframe(dataset.head())

# ==============================
# Feature Selection
# ==============================

X = dataset[['pickup_latitude', 'pickup_longitude']]

# ==============================
# Scaling
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# DBSCAN Experiments
# ==============================

eps_values = [0.2, 0.3, 0.5]
labels_list = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5, algorithm='ball_tree')
    labels = dbscan.fit_predict(X_scaled)
    labels_list.append(labels)

# ==============================
# Evaluation Function
# ==============================

def evaluate(labels):

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)

    if n_clusters > 1:
        X_non_noise = X_scaled[labels != -1]
        labels_non_noise = labels[labels != -1]
        score = silhouette_score(X_non_noise, labels_non_noise)
    else:
        score = None

    return n_clusters, n_noise, noise_ratio, score

# ==============================
# Visualization + Metrics
# ==============================

best_score = -1
best_eps = None

for i, labels in enumerate(labels_list):

    eps = eps_values[i]
    clusters, noise, ratio, score = evaluate(labels)

    st.subheader(f"Results for eps = {eps}")

    st.write("Number of Clusters:", clusters)
    st.write("Noise Points:", noise)
    st.write("Noise Ratio:", round(ratio, 3))

    if score:
        st.write("Silhouette Score:", round(score, 3))
        if score > best_score:
            best_score = score
            best_eps = eps
    else:
        st.write("Silhouette Score: Not Applicable")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.scatter(
        X['pickup_latitude'],
        X['pickup_longitude'],
        c=labels,
        s=20
    )

    ax.scatter(
        X.loc[labels == -1, 'pickup_latitude'],
        X.loc[labels == -1, 'pickup_longitude'],
        color='black',
        label='Noise'
    )

    ax.set_title(f"DBSCAN Clustering (eps={eps})")
    ax.set_xlabel("Pickup Latitude")
    ax.set_ylabel("Pickup Longitude")
    ax.legend()

    st.pyplot(fig)

# ==============================
# Best Model
# ==============================

st.success(f"🏆 Best eps value = {best_eps}")
