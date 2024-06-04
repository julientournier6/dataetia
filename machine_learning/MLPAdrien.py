import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chemin du fichier
file_path = 'classif_test_v2.xlsx'

# Charger les données depuis le fichier Excel
data = pd.read_excel(file_path)

# Préparer les données
X = data.drop(columns=['ID', 'bug type'])
y = data['bug type']

# Convertir les étiquettes cibles en valeurs numériques
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Sélectionner uniquement les colonnes numériques
X = X.select_dtypes(include=[float, int])

# Imputer les valeurs manquantes avec la moyenne
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définir les ensembles de meilleurs paramètres à tester
params_list = [
    {
        'hidden_layer_sizes': (50,),
        'activation': 'relu',
        'solver': 'sgd',
        'alpha': 3e-05,
        'learning_rate': 'constant',
        'learning_rate_init': 0.0045,
        'max_iter': 1500
    },
    {
        'hidden_layer_sizes': (50,),
        'activation': 'relu',
        'solver': 'sgd',
        'alpha': 3.5e-05,
        'learning_rate': 'constant',
        'learning_rate_init': 0.004,
        'max_iter': 1500
    },
    {
        'hidden_layer_sizes': (50,),
        'activation': 'relu',
        'solver': 'sgd',
        'alpha': 0.0001,
        'learning_rate': 'constant',
        'learning_rate_init': 0.0045,
        'max_iter': 1500
    }
]

# Stocker les résultats
results = []

# Tester chaque ensemble de paramètres
for params in params_list:
    # Initialiser le modèle MLPClassifier avec les paramètres actuels
    mlp = MLPClassifier(**params, random_state=42)
    
    # Validation croisée
    cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=5)
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    # Entraîner le modèle
    mlp.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = mlp.predict(X_test_scaled)
    
    # Précision sur l'ensemble de test
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Stocker les résultats
    results.append({
        'params': params,
        'cv_mean_accuracy': mean_cv_score,
        'cv_std_accuracy': std_cv_score,
        'test_accuracy': test_accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    })

# Afficher les résultats pour chaque ensemble de paramètres
for result in results:
    print(f"Parameters: {result['params']}")
    print(f"Cross-validation accuracy: {result['cv_mean_accuracy'] * 100:.2f}% (+/- {result['cv_std_accuracy'] * 100:.2f}%)")
    print(f"Test accuracy: {result['test_accuracy'] * 100:.2f}%")
    print("Classification report:")
    report_df = pd.DataFrame(result['classification_report']).transpose()
    print(report_df)
    print("\n" + "="*80 + "\n")

# Comparer les résultats
best_result = max(results, key=lambda x: x['test_accuracy'])
print(f"Best parameters: {best_result['params']}")
print(f"Best cross-validation accuracy: {best_result['cv_mean_accuracy'] * 100:.2f}% (+/- {best_result['cv_std_accuracy'] * 100:.2f}%)")
print(f"Best test accuracy: {best_result['test_accuracy'] * 100:.2f}%")
