import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================
# STEP 1: LOAD DATA
# ============================================
# Load the preprocessed CSV files containing HOG features
train_df = pd.read_csv("Images/Dataset/train_processed.csv")
test_df  = pd.read_csv("Images/Dataset/test_processed.csv")

# Combine train and test into one dataset
# ignore_index=True: Reset index to 0, 1, 2, ... for the combined dataframe
full_df = pd.concat([train_df, test_df], ignore_index=True)

# ============================================
# STEP 2: EXTRACT FEATURES AND LABELS
# ============================================
# Extract the true labels (ground truth): Apple=0, Cherry=1, Tomatoe=2
true_labels = full_df["label"].values  # Convert to numpy array

# Remove the label column to get only the features (HOG descriptors)
# These are the X values that K-means will cluster
features = full_df.drop("label", axis=1).values  # Shape: (n_samples, n_features)

print("Shape of features:", features.shape)
print(f"Number of samples: {features.shape[0]}")
print(f"Number of features per sample: {features.shape[1]}")

# ============================================
# IMPROVEMENT 1: FEATURE STANDARDIZATION
# ============================================
# WHY: HOG features have different scales/ranges
# Standardization makes all features have mean=0 and std=1
# This prevents features with large values from dominating the distance calculation
print("\n[IMPROVEMENT 1] Standardizing features...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print(f"✓ Features standardized (mean ≈ 0, std ≈ 1)")

# ============================================
# STEP 3: CREATE K-MEANS MODEL
# ============================================
n_clusters = 3  # We want 3 clusters (one for each fruit type)

# ORIGINAL (WEAK SETTINGS):
# kmeans = KMeans(n_clusters=3, random_state=42)

# IMPROVED K-MEANS WITH BETTER PARAMETERS:
kmeans = KMeans(
    n_clusters=n_clusters,           # Number of clusters to form
    init='k-means++',                # Smart initialization (better than random)
    n_init=50,                       # Run algorithm 50 times with different initializations
                                     # and pick the best result (default is only 10)
    max_iter=500,                    # Maximum iterations per run (default 300)
    tol=1e-4,                        # Convergence tolerance (default 1e-4)
    random_state=42,                 # For reproducibility
    algorithm='lloyd',               # Standard K-means algorithm
    verbose=0                        # Set to 1 to see progress
)

print("\n[IMPROVEMENT 2] Using optimized K-Means parameters:")
print(f"  - init='k-means++' (smart initialization)")
print(f"  - n_init=50 (try 50 different starting points)")
print(f"  - max_iter=500 (allow more iterations)")

# ============================================
# STEP 4: FIT K-MEANS AND GET CLUSTER ASSIGNMENTS
# ============================================
print("\n[STEP 4] Running K-Means clustering...")

# fit_predict does two things:
# 1. Learns the cluster centers from the data (training)
# 2. Assigns each sample to the nearest cluster (prediction)
# Result: array of cluster IDs [0, 1, 2] for each sample
cluster_labels = kmeans.fit_predict(features_scaled)  # USING SCALED FEATURES!

print("\n========= CLUSTER RESULTS =========")
for c in range(n_clusters):
    count = np.sum(cluster_labels == c)  # Count samples in this cluster
    percentage = (count / len(cluster_labels)) * 100
    print(f"Cluster {c}: {count} samples ({percentage:.1f}%)")

# ============================================
# STEP 5: MAP CLUSTERS TO TRUE LABELS
# ============================================
# CRITICAL CONCEPT: K-means doesn't know about "Apple", "Cherry", "Tomatoe"
# It only creates Cluster 0, 1, 2 based on feature similarity
# We need to figure out which cluster corresponds to which fruit

print("\n[STEP 5] Mapping clusters to fruit types...")
label_mapping = {}  # Dictionary: {cluster_id: true_label}

for cluster in range(n_clusters):
    # Get all samples that belong to this cluster
    mask = (cluster_labels == cluster)  # Boolean array: True where cluster matches
    
    # Safety check: skip if cluster is empty
    if np.sum(mask) == 0:
        continue
    
    # Get the true labels of all samples in this cluster
    true_labels_in_cluster = true_labels[mask]
    
    # Find the MOST COMMON true label in this cluster
    # Example: if cluster 0 has [0,0,0,1,0,2] → most common is 0 (Apple)
    # mode() returns the most frequent value
    # keepdims=True keeps the result as an array
    # .mode[0] extracts the actual value
    most_common = mode(true_labels_in_cluster, keepdims=True).mode[0]
    
    # Map this cluster to the most common label
    label_mapping[cluster] = most_common
    
    # Show what we found
    fruit_names = {0: "Apple", 1: "Cherry", 2: "Tomatoe"}
    print(f"  Cluster {cluster} → {fruit_names[most_common]}")
    
    # Show cluster composition for debugging
    unique_labels, counts = np.unique(true_labels_in_cluster, return_counts=True)
    for label, count in zip(unique_labels, counts):
        pct = (count / len(true_labels_in_cluster)) * 100
        print(f"    - {fruit_names[label]}: {count} ({pct:.1f}%)")

# ============================================
# STEP 6: CONVERT CLUSTER IDS TO PREDICTED LABELS
# ============================================
# Now convert cluster assignments [0,1,2] to fruit labels [0,1,2]
# Example: cluster_labels = [1, 0, 2, 1, 0]
#          label_mapping = {0: 2, 1: 0, 2: 1}  (cluster 0→Tomatoe, etc.)
#          predicted_labels = [0, 2, 1, 0, 2]

predicted_labels = np.array([label_mapping[c] for c in cluster_labels])

# ============================================
# STEP 7: CALCULATE ACCURACY
# ============================================
# accuracy_score compares two arrays element by element:
# - true_labels: what the fruit actually is
# - predicted_labels: what K-means predicted
# 
# Example:
#   true_labels =      [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
#   predicted_labels = [0, 1, 1, 0, 1, 2, 2, 1, 2, 0]
#   matches:           [✓, ✓, ✗, ✓, ✓, ✓, ✗, ✓, ✓, ✓]
#   accuracy = 8/10 = 0.80 = 80%

accuracy = accuracy_score(true_labels, predicted_labels)

print("\n========= FINAL RESULTS =========")
print(f"K-Means Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


