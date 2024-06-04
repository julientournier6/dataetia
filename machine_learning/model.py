import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

# Chemin du fichier
file_path = 'classif_test_v2.xlsx'

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

# Fonction pour évaluer les modèles avec cross-validation et afficher les rapports de classification
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Validation croisée
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    return report, accuracy, mean_cv_score, std_cv_score

# Entraîner et évaluer les modèles

# Régression logistique
log_reg = LogisticRegression(max_iter=200)
log_reg_report, log_reg_accuracy, log_reg_mean_cv, log_reg_std_cv = evaluate_model(log_reg, X_train_scaled, y_train, X_test_scaled, y_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn_report, knn_accuracy, knn_mean_cv, knn_std_cv = evaluate_model(knn, X_train_scaled, y_train, X_test_scaled, y_test)

# SVM
svm = SVC(C=1, gamma=0.1, kernel='rbf')
svm_report, svm_accuracy, svm_mean_cv, svm_std_cv = evaluate_model(svm, X_train_scaled, y_train, X_test_scaled, y_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=None)
rf_report, rf_accuracy, rf_mean_cv, rf_std_cv = evaluate_model(rf, X_train_scaled, y_train, X_test_scaled, y_test)

# MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
mlp_report, mlp_accuracy, mlp_mean_cv, mlp_std_cv = evaluate_model(mlp, X_train_scaled, y_train, X_test_scaled, y_test)

# Réseau de neurones avec Dropout et ajustement des paramètres
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_scaled, to_categorical(y_train), epochs=100, batch_size=32, validation_split=0.2, verbose=1)
loss, accuracy = nn_model.evaluate(X_test_scaled, to_categorical(y_test))
nn_report = {'loss': loss, 'accuracy': accuracy}
nn_accuracy = accuracy

# Enregistrer les résultats dans un fichier Excel
results = {
    'Logistic Regression': log_reg_report,
    'KNN': knn_report,
    'SVM': svm_report,
    'Random Forest': rf_report,
    'MLP': mlp_report,
    'Neural Network': nn_report
}
results_df = pd.DataFrame.from_dict(results, orient='index')
results_file_path = 'Machine_learning/apprentissage.xlsx'

# Ajouter les scores d'accuracy à results_df
accuracy_data = {
    'Model': ['Logistic Regression', 'KNN', 'SVM', 'Random Forest', 'MLP', 'Neural Network'],
    'Test Accuracy': [log_reg_accuracy, knn_accuracy, svm_accuracy, rf_accuracy, mlp_accuracy, nn_accuracy],
    'Cross-Validation Accuracy': [
        f"{log_reg_mean_cv * 100:.2f}% (+/- {log_reg_std_cv * 100:.2f}%)",
        f"{knn_mean_cv * 100:.2f}% (+/- {knn_std_cv * 100:.2f}%)",
        f"{svm_mean_cv * 100:.2f}% (+/- {svm_std_cv * 100:.2f}%)",
        f"{rf_mean_cv * 100:.2f}% (+/- {rf_std_cv * 100:.2f}%)",
        f"{mlp_mean_cv * 100:.2f}% (+/- {mlp_std_cv * 100:.2f}%)",
        "N/A"  # Cross-validation for Neural Network is not done
    ]
}
accuracy_df = pd.DataFrame(accuracy_data)
accuracy_df.set_index('Model', inplace=True)

# Enregistrer les résultats détaillés et les scores d'accuracy
with pd.ExcelWriter(results_file_path) as writer:
    results_df.to_excel(writer, sheet_name='Detailed Reports')
    accuracy_df.to_excel(writer, sheet_name='Accuracy Scores')

print(f"Results saved to {results_file_path}")

# Afficher les résultats de l'accuracy pour chaque modèle
for model, acc, cv_acc in zip(accuracy_data['Model'], accuracy_data['Test Accuracy'], accuracy_data['Cross-Validation Accuracy']):
    print(f"{model} Cross-Validation Accuracy: {cv_acc}")
    print(f"{model} Test Accuracy: {acc * 100:.2f}%\n")
