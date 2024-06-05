import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Charger les données
data = pd.read_excel('train/classif_test_v2.xlsx')

# Mapper les bug types aux nouvelles catégories
def map_bug_type(bug_type):
    if bug_type == 'Bee':
        return 'bee'
    elif bug_type == 'Bumblebee':
        return 'bumblebee'
    else:
        return 'others'

data['bug_category'] = data['bug type'].apply(map_bug_type)

# Préparer les données
features = data.drop(columns=['ID', 'bug type', 'species', 'bug_category'])
labels = data['bug_category']

# Imputer les valeurs manquantes si nécessaire
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Standardiser les données si nécessaire
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Modèle Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# Validation croisée stratifiée
skf = StratifiedKFold(n_splits=5)
scores = []

for train_index, test_index in skf.split(features_scaled, labels):
    X_train, X_test = features_scaled[train_index], features_scaled[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# Résultats
print(f'Accuracy Scores: {scores}')
print(f'Mean Accuracy: {np.mean(scores)}')
print(f'Standard Deviation: {np.std(scores)}')