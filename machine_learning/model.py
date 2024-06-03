import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Chemin du fichier
file_path = 'Machine learning/données.xlsx'

# Vérifier que le fichier existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist.")

# Charger les données depuis le fichier Excel
data = pd.read_excel(file_path)

# Afficher les colonnes pour vérifier les noms
print("Colonnes disponibles dans le DataFrame :")
print(data.columns)

# Préparer les données
X = data.drop(columns=['ID', 'bug type'])
y = data['bug type']

# Vérifier les tailles initiales
print("Taille initiale de X :", X.shape)
print("Taille initiale de y :", y.shape)

# Convertir les étiquettes cibles en valeurs numériques
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Vérifier la taille après encodage
print("Taille de y_encoded :", y_encoded.shape)

# Sélectionner uniquement les colonnes numériques
X = X.select_dtypes(include=[float, int])

# Imputer les valeurs manquantes avec la moyenne
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Vérifier la taille après imputation
print("Taille de X_imputed :", X_imputed.shape)

# Assurer l'alignement des indices
assert X_imputed.shape[0] == y_encoded.shape[0], "Le nombre d'échantillons dans X et y est incohérent après traitement."

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Régression logistique
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
log_reg_report = classification_report(y_test, y_pred, output_dict=True)

# SVM
svm = SVC()
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)

# Réseau de neurones
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_scaled, to_categorical(y_train), epochs=10, batch_size=32, validation_split=0.2)
loss, accuracy = nn_model.evaluate(X_test_scaled, to_categorical(y_test))
nn_report = {'loss': loss, 'accuracy': accuracy}

# Enregistrer les résultats dans un fichier Excel
results = {
    'Logistic Regression': log_reg_report,
    'SVM': svm_report,
    'Random Forest': rf_report,
    'Neural Network': nn_report
}
results_df = pd.DataFrame.from_dict(results, orient='index')
results_file_path = 'Machine learning/apprentissage.xlsx'
results_df.to_excel(results_file_path)

print(f"Results saved to {results_file_path}")
