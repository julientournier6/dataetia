import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, classification_report

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

# Définir les hyperparamètres à tester
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}

# Modèle Random Forest
rf = RandomForestClassifier(random_state=42)

# Grid Search avec validation croisée
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(features_train_scaled, labels_train)

# Afficher les meilleurs hyperparamètres
print(f"Best hyperparameters: {grid_search.best_params_}")

# Utiliser les meilleurs hyperparamètres pour entraîner le modèle final
best_rf = grid_search.best_estimator_

# Charger les nouvelles données
data_test = pd.read_excel('données2_v2.xlsx')

# Préparer les données de test de la même manière que les données d'entraînement
features_test = data_test.drop(columns=['ID'])
features_test_imputed = imputer.transform(features_test)
features_test_scaled = scaler.transform(features_test_imputed)

# Prédire les catégories des nouvelles images
predictions = best_rf.predict(features_test_scaled)

# Ajouter les prédictions au DataFrame des nouvelles données
data_test['bug_category'] = predictions

# Garder uniquement les colonnes ID et bug_category
result = data_test[['ID', 'bug_category']]

# Stocker les résultats dans un fichier Excel
output_file = 'machine_learning/noe_result.xlsx'
result.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")

# Évaluer le modèle sur les données d'entraînement avec cross-validation pour obtenir accuracy et recall
X_train, X_val, y_train, y_val = train_test_split(features_train_scaled, labels_train, test_size=0.2, random_state=42)
best_rf.fit(X_train, y_train)
y_pred_val = best_rf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred_val)
recall = recall_score(y_val, y_pred_val, average='weighted')

print(f"Accuracy: {accuracy:.2%}")
print(f"Recall: {recall:.2%}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val))