import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Charger les données depuis le fichier Excel
file_path = 'machine_learning/données.xlsx'
data = pd.read_excel(file_path)

# Créer une nouvelle colonne 'bug_category' pour les quatre catégories
def categorize_bug(bug_type):
    if bug_type == 'Bee':
        return 'Bee'
    elif bug_type == 'Bumblebee':
        return 'Bumblebee'
    elif bug_type == 'Butterfly':
        return 'Butterfly'
    else:
        return 'Others'

data['bug_category'] = data['bug type'].apply(categorize_bug)

# Séparer les caractéristiques (features)
X = data.drop(columns=['ID', 'bug type', 'species', 'bug_category'])  # Caractéristiques

# Gérer les valeurs manquantes en utilisant SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer l'agglomerative clustering avec 4 clusters
agglo = AgglomerativeClustering(n_clusters=4)
clusters = agglo.fit_predict(X_scaled)

# Évaluer le modèle de clustering avec le score de silhouette
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.2f}")