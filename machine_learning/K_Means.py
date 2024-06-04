import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Charger les données depuis le fichier Excel
data_path = '../classif_test_v2.xlsx'
data = pd.read_excel(data_path)

# Sélectionner les caractéristiques pour le clustering
features = data.drop(columns=['ID', 'bug type', 'species'])

# Gérer les valeurs manquantes
features = features.fillna(features.mean())

# Standardiser les données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Appliquer l'agglomérative clustering
best_score = -1
best_n_clusters = 0

for n_clusters in range(2, 10):  # Essayer différents nombres de clusters
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette score for {n_clusters} clusters: {score:.2f}")
    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters

print(f"Best number of clusters: {best_n_clusters} with a silhouette score of {best_score:.2f}")