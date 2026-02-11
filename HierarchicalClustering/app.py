import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.title("📰 News Article Hierarchical Clustering App")

# ---------------- LOAD DATA ----------------
dataset = pd.read_csv("all-data.csv", encoding='latin1')
dataset.columns = ['Sentiment', 'News']

st.subheader("Sample Articles")
st.write(dataset.head())

texts = dataset['News']

# ---------------- TF-IDF ----------------
st.info("Converting text to TF-IDF vectors...")

vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(texts)
X_dense = X.toarray()

st.success("TF-IDF completed!")

# ---------------- DENDROGRAM ----------------
st.subheader("Dendrogram (First 100 Articles)")

subset = X_dense[:100]

fig1, ax1 = plt.subplots(figsize=(12,6))
Z = sch.linkage(subset, method='ward')
sch.dendrogram(Z)
plt.axhline(y=1.5, color='r', linestyle='--')
plt.title("Dendrogram with Cut Line")
plt.xlabel("Articles")
plt.ylabel("Euclidean Distance")

st.pyplot(fig1)

# ---------------- CLUSTERING ----------------
n_clusters = st.slider("Select Number of Clusters", 2, 8, 5)

model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward'
)

labels = model.fit_predict(X_dense)

# ---------------- SILHOUETTE ----------------
score = silhouette_score(X_dense, labels)

st.subheader("Clustering Quality")
st.write("Silhouette Score:", round(score, 3))

# ---------------- PCA VISUALIZATION ----------------
st.subheader("PCA Cluster Visualization")

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_dense)

fig2, ax2 = plt.subplots(figsize=(8,6))

plt.scatter(
    X_reduced[:,0],
    X_reduced[:,1],
    c=labels,
    s=50
)

plt.title("Hierarchical Clustering (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

st.pyplot(fig2)

st.success("Hierarchical Clustering Completed Successfully!")
