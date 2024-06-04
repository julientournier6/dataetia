import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Charger les données
data = pd.read_excel('train/classif_test_v2.xlsx')

# Préparer les données
features = data.drop(columns=['ID', 'bug type', 'species'])
labels = data['bug type']

# Imputer les valeurs manquantes si nécessaire
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Standardiser les données si nécessaire
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Modèle Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# Validation croisée
scores = cross_val_score(model, features_scaled, labels, cv=5)  # cv=5 pour une validation croisée à 5 folds

# Résultats
print(f'Accuracy Scores: {scores}')
print(f'Mean Accuracy: {scores.mean()}')
print(f'Standard Deviation: {scores.std()}')