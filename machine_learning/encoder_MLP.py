import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
data_train = pd.read_excel('machine_learning/données.xlsx')

# Normaliser les noms de colonnes en supprimant les suffixes _x et _y et en les convertissant en minuscules
data_train.columns = [col.split('_')[0].strip().lower() for col in data_train.columns]

# Supprimer les colonnes inutiles pour l'entraînement
X = data_train.drop(columns=['id', 'bug type', 'species'])
y = data_train['bug type']

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduire la dimensionnalité avec PCA
pca = PCA(n_components=32)  # Vous pouvez ajuster ce nombre de composants
X_pca = pca.fit_transform(X_scaled)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Entraîner un MLP sur les caractéristiques réduites
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# Prédire et évaluer le modèle
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Imprimer les résultats
print("Accuracy: {:.2f}%".format(accuracy * 100))