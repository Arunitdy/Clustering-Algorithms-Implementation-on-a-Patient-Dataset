import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1️⃣ Load and preprocess data
# -----------------------------
df = pd.read_csv("./data/patient_dataset.csv")

# Encode Gender: Male=0, Female=1
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])

# Features for clustering
features = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# Derived feature: non-HDL cholesterol
df['non_HDL'] = df['Chol'] - df['HDL']
features.append('non_HDL')

# Standardize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# -----------------------------
# 2️⃣ Determine best k (optional if already done)
# -----------------------------
k_range = range(2, 9)
sil_scores = []
inertia = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)
    sil_scores.append(silhouette_score(df_scaled, cluster_labels))
    inertia.append(kmeans.inertia_)

# Plot silhouette and elbow
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(k_range, sil_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")

plt.subplot(1,2,2)
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Curve (Inertia) vs k")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()

# Choose best k
best_k = k_range[sil_scores.index(max(sil_scores))]
print("Best k based on silhouette score:", best_k)

# -----------------------------
# 3️⃣ Fit final K-Means
# -----------------------------
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(df_scaled)

# Cluster counts
print("\nCluster counts:\n", df['Cluster'].value_counts())

# -----------------------------
# 4️⃣ Cluster profiling
# -----------------------------
cluster_profile = df.groupby('Cluster')[features].median()
print("\nCluster Median Profile:\n", cluster_profile)

# -----------------------------
# 5️⃣ PCA 2D visualization
# -----------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df['PC1'] = pca_result[:,0]
df['PC2'] = pca_result[:,1]

print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='Set2', s=80)
plt.title('PCA 2D Scatter Plot by Cluster')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
plt.legend(title='Cluster')
plt.show()

# -----------------------------
# 6️⃣ Optional t-SNE visualization
# -----------------------------
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(df_scaled)
df['TSNE1'] = tsne_result[:,0]
df['TSNE2'] = tsne_result[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', data=df, palette='Set2', s=80)
plt.title('t-SNE 2D Scatter Plot by Cluster')
plt.show()

# -----------------------------
# 7️⃣ Save clustered data
# -----------------------------
df.to_csv("./data/patient_dataset_clustered.csv", index=False)
