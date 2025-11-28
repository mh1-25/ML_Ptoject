import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def load_images_from_folder(folder, img_size=(100, 100)):
    images = []
    filenames = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, filename)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                filenames.append(path)
    return np.array(images), filenames

def extract_color_features(images):
    """Extract color histogram features instead of raw pixels"""
    features = []
    for img in images:
        # Calculate histogram for each color channel
        hist_r = cv2.calcHist([img], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [32], [0, 256])
        
        # Normalize and concatenate
        hist_r = hist_r.flatten() / hist_r.sum()
        hist_g = hist_g.flatten() / hist_g.sum()
        hist_b = hist_b.flatten() / hist_b.sum()
        
        feature = np.concatenate([hist_r, hist_g, hist_b])
        features.append(feature)
    
    return np.array(features)

def kmeans_cluster(features, n_clusters=3):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, 
                             batch_size=1000, max_iter=100)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

def print_clusters(filenames, labels, n_clusters):
    print("\n========== CLUSTER RESULTS ==========\n")
    for cluster in range(n_clusters):
        cluster_files = [filenames[i] for i in range(len(labels)) if labels[i] == cluster]
        print(f"Cluster {cluster}: {len(cluster_files)} images")
        print("Example files:", cluster_files[:5])
        print("-------------------------------------")

if __name__ == "__main__":
    folder_path = "Images/Dataset/fruits-360-3-body-problem"
    
    print("Loading images...")
    images, filenames = load_images_from_folder(folder_path)
    
    print("Extracting color features...")
    features = extract_color_features(images)
    
    print("Running Mini-Batch K-means...")
    n_clusters = 3
    labels, kmeans = kmeans_cluster(features, n_clusters)
    
    print_clusters(filenames, labels, n_clusters)