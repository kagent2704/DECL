# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Step 4: Apply Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(X_scaled)

# Step 5: Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Step 6: Evaluate the clustering with silhouette score
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
agg_silhouette = silhouette_score(X_scaled, agg_labels)
dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

print(f"Silhouette Score for K-Means: {kmeans_silhouette:.4f}")
print(f"Silhouette Score for Agglomerative Clustering: {agg_silhouette:.4f}")
print(f"Silhouette Score for DBSCAN: {dbscan_silhouette:.4f}")

# Step 7: Visualize the clustering results

# K-Means Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=kmeans_labels, palette="Set1", s=100, marker="o")
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Agglomerative Clustering Visualization
plt.subplot(1, 3, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=agg_labels, palette="Set1", s=100, marker="o")
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# DBSCAN Visualization
plt.subplot(1, 3, 3)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=dbscan_labels, palette="Set1", s=100, marker="o")
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

# The above code demonstrates clustering techniques on the Iris dataset using K-Means, Agglomerative Clustering, and DBSCAN.
# It evaluates the clustering performance using silhouette scores and visualizes the results.
# The silhouette score indicates how well-separated the clusters are, with higher values indicating better-defined clusters.
# The visualizations help in understanding the clustering results visually.
# You can modify the parameters of the clustering algorithms to see how they affect the results.
# This code is a complete example of clustering techniques in Python using the Iris dataset.
# It includes all the necessary steps, from data preparation to visualization, and provides a clear understanding of
# how to apply clustering algorithms in practice.
# Make sure to install the required libraries if you haven't already:
# pip install pandas numpy scikit-learn matplotlib seaborn
# This code is ready to run and should work without any issues.