Clustering Patient Records

Student: Arun M
Roll No: 76
Section: R7B
CO Mapped: CO4
SDG Mapped: SDG 3 – Good Health and Well-being

1️⃣ Overview

This assignment performs unsupervised learning (K-Means clustering) on a dataset of patient clinical parameters. The goal is to identify hidden patient phenotypes and analyze potential health risks. PCA and t-SNE were applied for 2D visualization of the clusters.

Objectives:

Group similar patients using clustering.

Visualize clusters in 2D using dimensionality reduction.

Identify potential health-risk groups.

Suggest public health interventions aligned with SDG 3.

2️⃣ Data Description

Features used:

Demographics: Age, Gender

Vitals & Labs: Systolic/Diastolic BP, Urea, Creatinine, HbA1c, Cholesterol, Triglycerides, HDL, LDL, VLDL, BMI

Derived Feature: Non-HDL cholesterol = total cholesterol − HDL

Preprocessing:

Encode Gender (Male=0, Female=1)

Scale all features using StandardScaler

Add derived feature: non-HDL cholesterol

Code snippet:

# Data preprocessing and feature scaling
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['non_HDL'] = df['Chol'] - df['HDL']
features.append('non_HDL')
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])


Output Screenshot Placeholder:


3️⃣ Clustering Implementation

Algorithm: K-Means
Steps:

Run K-Means for k = 2 to 8

Compute silhouette scores

Plot elbow curve (inertia)

Choose best k based on silhouette score

Code snippet:

k_range = range(2, 9)
sil_scores = []
inertia = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)
    sil_scores.append(silhouette_score(df_scaled, cluster_labels))
    inertia.append(kmeans.inertia_)


Silhouette & Elbow Curve Screenshot Placeholder:


Selected k:

Best k based on silhouette score: 2


Final K-Means Fit:

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(df_scaled)
print(df['Cluster'].value_counts())


Cluster Counts Screenshot Placeholder:


4️⃣ Cluster Profiling

Code snippet:

cluster_profile = df.groupby('Cluster')[features].median()
print(cluster_profile)


Cluster Median Profile Screenshot Placeholder:


Interpretation Table:

Cluster	Phenotype	Key Features	Potential Health Risks
0	Lower-risk / active	Lower BMI, glucose, BP, higher HDL	Reinforce healthy behavior
1	Metabolic-risk	High BMI, glucose, TG, low HDL, high non-HDL	Risk of diabetes, dyslipidemia; lifestyle intervention recommended

Insights:

Metabolic-risk cluster → Early lifestyle intervention

Hypertensive cluster → BP monitoring & antihypertensive adherence

Older mixed-risk → Smoking cessation & statin evaluation

Lower-risk/active → Reinforce healthy behaviors

5️⃣ Dimensionality Reduction & Visualization
PCA (2D)

Code snippet:

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df['PC1'] = pca_result[:,0]
df['PC2'] = pca_result[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='Set2', s=80)
plt.show()


Variance Explained:

PC1: 25%

PC2: 17.6%

PCA Scatter Plot Screenshot Placeholder:


Interpretation: PCA shows partial separation along metabolic vs hypertensive dimensions.

Optional t-SNE

Code snippet:

tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(df_scaled)
df['TSNE1'] = tsne_result[:,0]
df['TSNE2'] = tsne_result[:,1]


t-SNE Scatter Plot Screenshot Placeholder:


6️⃣ Public Health Relevance / SDG 3

Clustering helps target interventions and optimize resources, supporting SDG 3:

Targeted Screening: HbA1c, lipid, BP for metabolic and hypertensive clusters

Lifestyle Interventions: Nutrition, smoking cessation, physical activity programs

Resource Optimization: Staff, devices, and medication allocation

Preventive Care: Early identification reduces NCD mortality and improves health coverage

7️⃣ Submission Details

Python Code: All preprocessing, K-Means, PCA, t-SNE

Plots: Silhouette, elbow curve, PCA scatter, t-SNE scatter

Tables: Cluster median profile, cluster counts

Outputs: Screenshots included in ./screenshots/ folder

Cluster-labeled CSV:

df.to_csv("./data/patient_dataset_clustered.csv", index=False)


Instructions to Include Screenshots:

Create a folder called screenshots in your submission directory.

Save each output plot or table as .png inside this folder.

Name them clearly as per placeholders (scaled_data.png, silhouette_elbow.png, cluster_counts.png, etc.)

Ensure the README references these files correctly.