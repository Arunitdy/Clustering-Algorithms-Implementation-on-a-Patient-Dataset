import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("data/patient_dataset.csv")

print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# ------------------------------
# 2. Preprocessing
# ------------------------------
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
df["Class"] = encoder.fit_transform(df["Class"])  # Keep for later comparison

# Features & labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 3. Apply Clustering
# ------------------------------
# KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=2)
df["Agglo_Cluster"] = agg.fit_predict(X_scaled)

# ------------------------------
# 4. Visualization
# ------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["AGE"], y=df["Chol"], hue=df["KMeans_Cluster"], palette="Set1")
plt.title("KMeans Clustering (AGE vs Chol)")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.savefig("figures/kmeans_clusters.png")  # Save plot
plt.show()

# ------------------------------
# 5. Evaluation
# ------------------------------
print("\n--- Evaluation ---")
print("KMeans Accuracy:", accuracy_score(y, df["KMeans_Cluster"]))
print("Agglomerative Accuracy:", accuracy_score(y, df["Agglo_Cluster"]))

print("\nKMeans Confusion Matrix:\n", confusion_matrix(y, df["KMeans_Cluster"]))
print("\nAgglomerative Confusion Matrix:\n", confusion_matrix(y, df["Agglo_Cluster"]))
