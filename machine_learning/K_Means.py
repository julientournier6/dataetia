import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Charger les données depuis le fichier Excel
data_path = 'machine_learning/données.xlsx'
data = pd.read_excel(data_path)

# Sélectionner les caractéristiques pour le clustering
features = data.drop(columns=['ID', 'bug type', 'species'])

# Gérer les valeurs manquantes
features = features.fillna(features.mean())

# Standardiser les données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Essayer différents nombres de clusters pour voir lequel donne le meilleur score de silhouette
best_score = -1
best_k = 0
for k in range(2, 10):  # Essayez différentes valeurs de k
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette score for {k} clusters: {score:.2f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best number of clusters: {best_k} with a silhouette score of {best_score:.2f}")