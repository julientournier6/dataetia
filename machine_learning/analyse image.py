import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Charger les nouvelles données
new_data_path = 'données2_v2.xlsx'
new_data = pd.read_excel(new_data_path)

# Charger les vraies valeurs des types d'insectes
true_data_path = 'classif 251-347.xlsx'
true_data = pd.read_excel(true_data_path)

# Préparer les données
X_new = new_data.drop(columns=['ID'])
true_labels = true_data['Bug Type']

# Sélectionner uniquement les colonnes numériques
X_new = X_new.select_dtypes(include=[float, int])

# Imputer les valeurs manquantes avec la moyenne (si nécessaire)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_new_imputed = imputer.fit_transform(X_new)

# Normaliser les données
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new_imputed)

# Charger le modèle entraîné et le label encoder
with open('model.pkl', 'rb') as file:
    model, le = pickle.load(file)

# Ajouter les classes manquantes aux labels
all_labels = np.concatenate((true_labels, le.classes_))
le.fit(all_labels)

# Prédire les types d'insectes
predictions = model.predict(X_new_scaled)
predicted_labels = le.inverse_transform(predictions)

# Ajouter les prédictions au DataFrame original
new_data['predicted_bug_type'] = predicted_labels

# Convertir les vraies étiquettes en valeurs numériques
true_labels_encoded = le.transform(true_labels)

# Calculer le taux de bonne réponse
accuracy = accuracy_score(true_labels_encoded, predictions)

print(f"Taux de bonne réponse: {accuracy * 100:.2f}%")

# Afficher et enregistrer les résultats
print("Predictions counts:")
print(new_data['predicted_bug_type'].value_counts())

output_file_path = 'predictions.xlsx'
new_data.to_excel(output_file_path, index=False)

print(f"Predicted 'bug type' saved to {output_file_path}")
