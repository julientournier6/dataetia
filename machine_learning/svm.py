import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, classification_report

# Charger les données d'entraînement
data_train = pd.read_excel('train/classif_test_v2.xlsx')

# Charger les nouvelles données manuelles
manual_data = pd.read_excel('classif 251-347.xlsx')

# Mapper les bug types aux nouvelles catégories
def map_bug_type(bug_type):
    if bug_type == 'Bee':
        return 'bee'
    elif bug_type == 'Bumblebee':
        return 'bumblebee'
    else:
        return 'others'

data_train['bug_category'] = data_train['bug type'].apply(map_bug_type)
manual_data['bug_category'] = manual_data['BugType'].apply(map_bug_type)

# Préparer les données d'entraînement
features_train = data_train.drop(columns=['ID', 'bug type', 'species', 'bug_category'])
labels_train = data_train['bug_category']

# Imputer les valeurs manquantes si nécessaire
imputer = SimpleImputer(strategy='mean')
features_train_imputed = imputer.fit_transform(features_train)

# Standardiser les données si nécessaire
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train_imputed)

# Préparer les données de validation manuelle
features_manual_val = manual_data.drop(columns=['ID', 'BugType'])
labels_manual_val = manual_data['bug_category']
features_manual_val_imputed = imputer.transform(features_manual_val)
features_manual_val_scaled = scaler.transform(features_manual_val_imputed)

# Définir les hyperparamètres à tester
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Modèle SVM
svm = SVC()

# Grid Search avec validation croisée
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(features_train_scaled, labels_train)

# Afficher les meilleurs hyperparamètres
print(f"Best hyperparameters: {grid_search.best_params_}")

# Utiliser les meilleurs hyperparamètres pour entraîner le modèle final
best_svm = grid_search.best_estimator_

# Évaluer le modèle sur les données de validation manuelle
y_pred_manual_val = best_svm.predict(features_manual_val_scaled)

manual_accuracy = accuracy_score(labels_manual_val, y_pred_manual_val)
manual_recall = recall_score(labels_manual_val, y_pred_manual_val, average='weighted')

print(f"Manual Validation Accuracy: {manual_accuracy:.2%}")
print(f"Manual Validation Recall: {manual_recall:.2%}")
print("\nManual Validation Classification Report:")
print(classification_report(labels_manual_val, y_pred_manual_val))

# Charger les nouvelles données à classer
data_test = pd.read_excel('données2_v2.xlsx')

# Préparer les données de test de la même manière que les données d'entraînement
features_test = data_test.drop(columns=['ID'])
features_test_imputed = imputer.transform(features_test)
features_test_scaled = scaler.transform(features_test_imputed)

# Prédire les catégories des nouvelles images avec le modèle optimisé
predictions = best_svm.predict(features_test_scaled)

# Ajouter les prédictions au DataFrame des nouvelles données
data_test['bug_category'] = predictions

# Garder uniquement les colonnes ID et bug_category
result = data_test[['ID', 'bug_category']]

# Stocker les résultats dans un fichier Excel
output_file = 'machine_learning/noe_result_svm.xlsx'
result.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")