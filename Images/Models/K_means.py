import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

train_df = pd.read_csv("Images/Dataset/train_processed.csv")
test_df  = pd.read_csv("Images/Dataset/test_processed.csv")

full_df = pd.concat([train_df, test_df], ignore_index=True)

true_labels = full_df["label"].values
features_df = full_df.drop("label", axis=1)  # KEEP AS DATAFRAME

numeric_columns = features_df.columns  # Only numeric columns

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numeric_columns)
    ]
)

kmeans_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', MiniBatchKMeans(n_clusters=3, random_state=42, max_iter=100, batch_size=1400)) #1400 => 49.30
])
4
kmeans_pipeline.fit(features_df)
cluster_labels = kmeans_pipeline['kmeans'].labels_

for c in range(3):
    count = np.sum(cluster_labels == c)
    percentage = (count / len(cluster_labels)) * 100
    print(f"Cluster {c}: {count} samples ({percentage:.1f}%)")

label_mapping = {}
for cluster in range(3):
    mask = (cluster_labels == cluster)
    if np.sum(mask) == 0:
        continue
    true_labels_in_cluster = true_labels[mask]
    most_common = mode(true_labels_in_cluster, keepdims=True).mode[0]
    label_mapping[cluster] = most_common

predicted_labels = np.array([label_mapping[c] for c in cluster_labels])

accuracy = accuracy_score(true_labels, predicted_labels)
print("\n========= FINAL RESULTS =========")
print(f"K-Means Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
