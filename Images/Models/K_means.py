import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import mode
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ===================== LOAD DATA =====================
train_df = pd.read_csv("ML_Ptoject/Images/Dataset/train_processed.csv")
test_df  = pd.read_csv("ML_Ptoject/Images/Dataset/test_processed.csv")

true_labels = train_df["label"].values
features_df = train_df.drop("label", axis=1)

numeric_columns = features_df.columns

# ===================== PREPROCESSING =====================
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numeric_columns)
    ]
)

import umap

# ===================== K-MEANS PIPELINE =====================
kmeans_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('umap', umap.UMAP(n_components=2, random_state=42)),
    ('kmeans', KMeans(n_clusters=7, random_state=42, n_init='auto'))
])

# ===================== TRAIN =====================
kmeans_pipeline.fit(features_df)

cluster_labels = kmeans_pipeline.named_steps['kmeans'].labels_

# ===================== CLUSTER DISTRIBUTION =====================
for c in range(7):
    count = np.sum(cluster_labels == c)
    percentage = (count / len(cluster_labels)) * 100
    print(f"Cluster {c}: {count} samples ({percentage:.1f}%)")

# ===================== CLUSTER → LABEL MAPPING =====================
label_mapping = {}

for c in range(7):
    mask = cluster_labels == c
    if np.sum(mask) == 0:
        continue

    true_labels_in_cluster = true_labels[mask]
    label_mapping[c] = mode(true_labels_in_cluster, keepdims=True).mode[0]

# ===================== PREDICTION =====================
predicted_labels = np.array([label_mapping[c] for c in cluster_labels])

# ===================== EVALUATION =====================
ari = adjusted_rand_score(true_labels, cluster_labels)
nmi = normalized_mutual_info_score(true_labels, cluster_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

print("\n========= FINAL RESULTS =========")
print(f"Accuracy (after mapping): {accuracy*100:.2f}%")
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")