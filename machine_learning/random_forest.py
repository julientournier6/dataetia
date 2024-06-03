import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer


# Charger les données depuis le fichier Excel
file_path = 'machine_learning/données.xlsx'
data = pd.read_excel(file_path)

# Afficher un aperçu des données
print(data.head())
print(data.dtypes)

# Séparer les caractéristiques (features) et l'étiquette (target)
X = data.drop(columns=['ID', 'bug type', 'species'])  # Caractéristiques
y = data['bug type']  # Étiquette

# Gérer les valeurs manquantes en utilisant SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardiser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer le modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

# Afficher les résultats
print("Confusion Matrix (en pourcentage):")
conf_matrix_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
print(conf_matrix_percentage)

print("\nClassification Report (en pourcentage):")
for label, metrics in class_report.items():
    if isinstance(metrics, dict):
        print(f"Classe {label}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value * 100:.2f}%")
    else:
        print(f"{label}: {metrics * 100:.2f}%")

print(f"\nAccuracy Score: {accuracy * 100:.2f}%")