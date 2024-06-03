import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

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

# Afficher un aperçu des données
print(data.head())
print(data.dtypes)

# Séparer les caractéristiques (features) et l'étiquette (target)
X = data.drop(columns=['ID', 'bug type', 'species', 'bug_category'])  # Caractéristiques
y = data['bug_category']  # Étiquette

# Gérer les valeurs manquantes en utilisant SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Créer le modèle k-Means
kmeans = KMeans(n_clusters=4, random_state=42)

# Entraîner le modèle
kmeans.fit(X_scaled)

# Prédictions de clusters
clusters = kmeans.predict(X_scaled)

# Mapping des clusters aux catégories
cluster_map = {}
for cluster in range(4):
    mask = (clusters == cluster)
    most_common = y[mask].mode().values[0]
    cluster_map[cluster] = most_common

# Traduire les clusters en catégories
y_pred = pd.Series(clusters).map(cluster_map)

# Évaluer les performances du modèle
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)
accuracy = accuracy_score(y, y_pred)

# Afficher les résultats
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

print(f"\nAccuracy Score: {accuracy * 100:.2f}%")