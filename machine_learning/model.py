import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Régression logistique avec GridSearchCV
log_reg = LogisticRegression(max_iter=200)
param_grid_log_reg = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
    'penalty': ['l2', 'none']
}
grid_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5, scoring='accuracy', n_jobs=-1)
grid_log_reg.fit(X_train_scaled, y_train)

# Meilleurs paramètres trouvés par GridSearchCV
best_params_log_reg = grid_log_reg.best_params_
print(f"Best parameters for Logistic Regression: {best_params_log_reg}")

# Prédictions et rapport de classification avec les meilleurs paramètres
y_pred_log_reg = grid_log_reg.predict(X_test_scaled)
log_reg_report = classification_report(y_test, y_pred_log_reg, output_dict=True)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)

# K-Nearest Neighbors avec GridSearchCV
knn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train)

# Meilleurs paramètres trouvés par GridSearchCV
best_params_knn = grid_knn.best_params_
print(f"Best parameters for KNN: {best_params_knn}")

# Prédictions et rapport de classification avec les meilleurs paramètres
y_pred_knn = grid_knn.predict(X_test_scaled)
knn_report = classification_report(y_test, y_pred_knn, output_dict=True)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

# SVM avec GridSearchCV
svm = SVC()
param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5)
grid_svm.fit(X_train_scaled, y_train)
y_pred_svm = grid_svm.predict(X_test_scaled)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

# Random Forest avec GridSearchCV
rf = RandomForestClassifier()
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_rf.fit(X_train_scaled, y_train)
y_pred_rf = grid_rf.predict(X_test_scaled)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# MLPClassifier avec GridSearchCV
mlp = MLPClassifier(max_iter=500)
param_grid_mlp = {
    'hidden_layer_sizes': [(50, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.1]
}
grid_mlp = GridSearchCV(mlp, param_grid_mlp, cv=5, scoring='accuracy', n_jobs=-1)
grid_mlp.fit(X_train_scaled, y_train)

# Meilleurs paramètres trouvés par GridSearchCV
best_params_mlp = grid_mlp.best_params_
print(f"Best parameters for MLP: {best_params_mlp}")

# Prédictions et rapport de classification avec les meilleurs paramètres
y_pred_mlp = grid_mlp.predict(X_test_scaled)
mlp_report = classification_report(y_test, y_pred_mlp, output_dict=True)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

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
    'Accuracy': [log_reg_accuracy, knn_accuracy, svm_accuracy, rf_accuracy, mlp_accuracy, nn_accuracy]
}
accuracy_df = pd.DataFrame(accuracy_data)
accuracy_df.set_index('Model', inplace=True)

# Enregistrer les résultats détaillés et les scores d'accuracy
with pd.ExcelWriter(results_file_path) as writer:
    results_df.to_excel(writer, sheet_name='Detailed Reports')
    accuracy_df.to_excel(writer, sheet_name='Accuracy Scores')

print(f"Results saved to {results_file_path}")

# Afficher les résultats de l'accuracy pour chaque modèle
print(f"Logistic Regression Accuracy: {log_reg_accuracy * 100:.2f}%")
print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f"MLP Accuracy: {mlp_accuracy * 100:.2f}%")
print(f"Neural Network Accuracy: {nn_accuracy * 100:.2f}%")
