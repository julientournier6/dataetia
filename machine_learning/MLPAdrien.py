import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Chemin des fichiers
file_path_train = 'classif_test_v2.xlsx'
file_path_new_data = 'données2_v2.xlsx'
output_file_path = 'predictions.xlsx'

# Charger les données d'entraînement depuis le fichier Excel
data = pd.read_excel(file_path_train)

# Réduire les catégories à "Bee", "Bumblebee" et "other"
data['bug type'] = data['bug type'].apply(lambda x: x if x in ['Bee', 'Bumblebee'] else 'other')

# Afficher les valeurs uniques de "bug type" avant transformation
print("Valeurs uniques de 'bug type' avant transformation :")
print(data['bug type'].value_counts())

# Préparer les données d'entraînement
X_train = data.drop(columns=['ID', 'bug type'])
y_train = data['bug type']

# Convertir les étiquettes cibles en valeurs numériques
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Sélectionner uniquement les colonnes numériques
X_train = X_train.select_dtypes(include=[float, int])

# Imputer les valeurs manquantes avec la moyenne
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Normaliser les données d'entraînement
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Appliquer SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)

# Définir les meilleurs hyperparamètres trouvés précédemment
best_params = {
    'hidden_layer_sizes': (50,),
    'activation': 'relu',
    'solver': 'sgd',
    'alpha': 0.0001,
    'learning_rate': 'constant',
    'learning_rate_init': 0.0045,
    'max_iter': 1500
}

# Initialiser et entraîner le modèle MLPClassifier avec les meilleurs paramètres
best_mlp = MLPClassifier(**best_params, random_state=42)
best_mlp.fit(X_train_resampled, y_train_resampled)

# Évaluer le modèle avec les données de test
X_train, X_test, y_train, y_test = train_test_split(X_train_imputed, y_train_encoded, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)
y_pred = best_mlp.predict(X_test_scaled)
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Classification Report on Test Data:")
print(report)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Prédire les "bug type" pour l'ensemble des données d'entraînement
y_train_pred = best_mlp.predict(X_train_scaled)
y_train_pred_labels = label_encoder.inverse_transform(y_train_pred)

# Ajouter les prédictions au DataFrame original
data['predicted_bug_type'] = y_train_pred_labels

# Compter le nombre de chaque catégorie prédite dans l'ensemble d'entraînement
train_prediction_counts = data['predicted_bug_type'].value_counts()
print("Training Predictions counts:")
print(train_prediction_counts)

# Charger le nouveau jeu de données
new_data = pd.read_excel(file_path_new_data)

# Préparer les données (supposant que la structure est similaire à l'ancien fichier)
X_new = new_data.drop(columns=['ID'])

# Sélectionner uniquement les colonnes numériques
X_new = X_new.select_dtypes(include=[float, int])

# Imputer et normaliser les données du nouveau jeu de données
X_new_imputed = imputer.transform(X_new)
X_new_scaled = scaler.transform(X_new_imputed)

# Appliquer le modèle sur le nouveau jeu de données pour prédire les "bug type" de manière indépendante
y_new_pred = best_mlp.predict(X_new_scaled)

# Convertir les étiquettes prédictes en leurs noms originaux
y_new_pred_labels = label_encoder.inverse_transform(y_new_pred)

# Ajouter les prédictions au DataFrame original
new_data['predicted_bug_type'] = y_new_pred_labels

# Compter le nombre de chaque catégorie prédite dans le nouveau jeu de données
prediction_counts = new_data['predicted_bug_type'].value_counts()
print("Predictions counts:")
print(prediction_counts)

# Enregistrer les résultats dans un nouveau fichier Excel
new_data.to_excel(output_file_path, index=False)

print(f"Predicted 'bug type' saved to {output_file_path}")