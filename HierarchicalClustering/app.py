import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.title("📰 News Article Hierarchical Clustering")

# Load dataset
dataset = pd.read_csv("all-data.csv", encoding="latin1")
dataset.columns = ["Sentiment", "Text"]

st.subheader("Sample Data")
st.write(dataset.head())

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=500)
x = tfidf.fit_transform(dataset["Text"]).toarray()

st.success("Text converted to TF-IDF vectors")

# ---------------- DENDROGRAM ----------------
st.subheader("Dendrogram (First 200 Articles)")

fig1, ax1 = plt.subplots(figsize=(8,5))
sch.dendrogram(sch.linkage(x[:200], method="ward"))
plt.axhline(y=180, color='r', linestyle='--')
plt.title("Dendrogram")
plt.xlabel("Articles")
plt.ylabel("Distance")
st.pyplot(fig1)

# ---------------- CLUSTERING ----------------
n_clusters = st.slider("Select Number of Clusters", 2, 6, 3)

hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
y_hc = hc.fit_predict(x)

# ---------------- SILHOUETTE ----------------
score = silhouette_score(x, y_hc)

st.subheader("Clustering Evaluation")
st.write("Silhouette Score:", round(score, 3))

# ---------------- PCA VISUALIZATION ----------------
st.subheader("PCA Cluster Visualization")

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

fig2, ax2 = plt.subplots()

for i in range(n_clusters):
    plt.scatter(x_pca[y_hc==i,0], x_pca[y_hc==i,1], s=60, label=f'Cluster {i+1}')

plt.title('Hierarchical Clustering (Text Data)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

st.pyplot(fig2)

st.success("Hierarchical Clustering Completed!")
