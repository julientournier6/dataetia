import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pickle

# Charger les données depuis le fichier Excel
file_path = 'classif_test_v2.xlsx'
data = pd.read_excel(file_path)

# Préparer les données
X = data.drop(columns=['ID', 'bug type'])
y = data['bug type']

# Réduire les catégories à "Bee", "Bumblebee" et "other"
y = y.apply(lambda x: x if x in ['Bee', 'Bumblebee'] else 'other')

# Convertir les étiquettes cibles en valeurs numériques
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Ajouter "other" aux classes du LabelEncoder si nécessaire
if 'other' not in label_encoder.classes_:
    label_encoder.classes_ = np.append(label_encoder.classes_, 'other')

# Mapper les classes rares à "other"
label_counts = pd.Series(y_encoded).value_counts()
total_samples = len(y_encoded)
threshold = 0.05 * total_samples  # Seuil de 5%
other_label = label_encoder.transform(['other'])[0]

y_encoded = np.array([label if label_counts[label] >= threshold else other_label for label in y_encoded])

# Sélectionner uniquement les colonnes numériques
X = X.select_dtypes(include=[float, int])

# Imputer les valeurs manquantes avec la moyenne
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Définir le modèle XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Définir la grille de paramètres pour l'ajustement des hyperparamètres
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Configurer la recherche randomisée avec validation croisée
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, cv=StratifiedKFold(n_splits=5), n_jobs=-1, scoring='accuracy')

# Entraîner le modèle avec la recherche randomisée
random_search.fit(X_train, y_train)

# Obtenir les meilleurs paramètres
best_params = random_search.best_params_
print(f"Best parameters found: {best_params}")

# Entraîner le modèle final avec les meilleurs paramètres
best_model = random_search.best_estimator_

# Sauvegarder le modèle entraîné et le label encoder
with open('model.pkl', 'wb') as file:
    pickle.dump((best_model, label_encoder), file)

print("Model and LabelEncoder trained and saved as model.pkl with best parameters")

# Évaluer le modèle
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Classification Report on Test Data:")
print(report)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
