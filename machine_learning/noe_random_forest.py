import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Charger les données d'entraînement
data_train = pd.read_excel('train/classif_test_v2.xlsx')

# Mapper les bug types aux nouvelles catégories
def map_bug_type(bug_type):
    if bug_type == 'Bee':
        return 'bee'
    elif bug_type == 'Bumblebee':
        return 'bumblebee'
    else:
        return 'others'

data_train['bug_category'] = data_train['bug type'].apply(map_bug_type)

# Préparer les données d'entraînement
features_train = data_train.drop(columns=['ID', 'bug type', 'species', 'bug_category'])
labels_train = data_train['bug_category']

# Imputer les valeurs manquantes si nécessaire
imputer = SimpleImputer(strategy='mean')
features_train_imputed = imputer.fit_transform(features_train)

# Standardiser les données si nécessaire
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train_imputed)

# Modèle Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# Validation croisée pour évaluer l'accuracy
scores = cross_val_score(model, features_train_scaled, labels_train, cv=5, scoring='accuracy')
mean_accuracy = scores.mean() * 100
print(f"Mean Accuracy: {mean_accuracy:.2f}%")

# Entraîner le modèle sur les données d'entraînement complètes
model.fit(features_train_scaled, labels_train)

# Charger les nouvelles données
data_test = pd.read_excel('données2_v2.xlsx')

# Préparer les données de test de la même manière que les données d'entraînement
features_test = data_test.drop(columns=['ID'])
features_test_imputed = imputer.transform(features_test)
features_test_scaled = scaler.transform(features_test_imputed)

# Prédire les catégories des nouvelles images
predictions = model.predict(features_test_scaled)

# Ajouter les prédictions au DataFrame des nouvelles données
data_test['bug_category'] = predictions

# Garder uniquement les colonnes ID et bug_category
result = data_test[['ID', 'bug_category']]

# Stocker les résultats dans un fichier Excel
output_file = 'machine_learning/noe_result_v2.xlsx'
result.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")