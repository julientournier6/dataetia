import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Charger les données
data_train = pd.read_excel('../classif_test_v2.xlsx')
data_test = pd.read_excel('machine_learning/données2.xlsx')

# Standardiser les noms de colonnes en supprimant les suffixes _x et _y
data_train.columns = [col.split('_')[0].lower() for col in data_train.columns]
data_test.columns = [col.lower() for col in data_test.columns]

# Supprimer les colonnes inutiles pour l'entraînement
X_train = data_train.drop(columns=['id', 'bug type', 'species'])
y_train = data_train['bug type']

# Vérifier les colonnes disponibles dans data_test
print("Colonnes de data_test avant alignement :")
print(data_test.columns)

# Assurer que les colonnes de X_test correspondent à celles de X_train
X_test = data_test[X_train.columns]

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construire l'auto-encodeur avec PCA (comme une alternative simple)
pca = PCA(n_components=32)  # Vous pouvez ajuster cette valeur
X_train_encoded = pca.fit_transform(X_train_scaled)
X_test_encoded = pca.transform(X_test_scaled)

# Entraîner un modèle supervisé sur les caractéristiques encodées
# Utiliser la régression logistique pour cette tâche
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_encoded, y_train)

# Prédire les labels pour le nouveau dataset
y_test_pred = log_reg.predict(X_test_encoded)

# Ajouter les prédictions au DataFrame de test
data_test['Predicted Bug Type'] = y_test_pred

# Enregistrer les prédictions dans un fichier Excel
data_test.to_excel('machine_learning/predicted_données2.xlsx', index=False)

print("Prédictions enregistrées dans 'machine_learning/predicted_données2.xlsx'")