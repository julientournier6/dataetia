import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Charger les données depuis le fichier Excel
file_path = 'classif_test_v2.xlsx'
data = pd.read_excel(file_path)

# Créer une nouvelle colonne 'bug_category' pour les quatre catégories
def categorize_bug(bug_type):
    if bug_type == 'Bee':
        return 'Bee'
    elif bug_type == 'Bumblebee':
        return 'Bumblebee'
    elif bug_type == 'Butterfly':
        return 'Butterfly'
    else:
        return 'Others'

data['bug_category'] = data['bug type'].apply(categorize_bug)

# Vérifier et afficher les colonnes du DataFrame
print(data.columns)

# Vérifier et créer les colonnes manquantes
cols_to_drop = ['id', 'bug type', 'species', 'bug_category']
for col in cols_to_drop:
    if col not in data.columns:
        data[col] = 'missing'

# Afficher un aperçu des données et les noms de colonnes
print(data.head())
print(data.columns)

# Séparer les caractéristiques (features) et l'étiquette (target)
existing_cols_to_drop = [col for col in cols_to_drop if col in data.columns]
X = data.drop(columns=existing_cols_to_drop)  # Caractéristiques
y = data['bug_category']  # Étiquette

# Gérer les valeurs manquantes en utilisant SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardiser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer et configurer le modèle Random Forest avec des hyperparamètres optimisés
model = RandomForestClassifier(
    n_estimators=1000,  # Augmenter le nombre d'arbres
    max_features='sqrt',  # Utiliser la racine carrée du nombre de caractéristiques
    max_depth=10,  # Limiter la profondeur des arbres
    min_samples_split=20,  # Augmenter le nombre minimum d'échantillons pour diviser un nœud
    min_samples_leaf=5,  # Augmenter le nombre minimum d'échantillons dans un nœud terminal
    bootstrap=True,  # Utiliser le bootstrap
    random_state=42
)

# Fonction pour évaluer un modèle avec validation croisée
def evaluate_model(model, X, y, cv_splits=5):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    return cv_scores.mean(), cv_scores.std()

# Évaluer le modèle Random Forest avec validation croisée
mean_rf, std_rf = evaluate_model(model, X_train, y_train)
print("Random Forest Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)".format(mean_rf * 100, std_rf * 100))

# Entraîner le modèle avec les meilleurs paramètres
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Afficher les résultats
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

print(f"\nRandom Forest Test Accuracy: {accuracy * 100:.2f}%")

# Charger les nouvelles données depuis le fichier Excel
new_data_path = 'données2_v2.xlsx'
data_new = pd.read_excel(new_data_path, engine='openpyxl')  # Spécifier le moteur openpyxl

# Afficher un aperçu des nouvelles données
print(data_new.head())
print(data_new.columns)

# Préparer les nouvelles données de la même manière que les données d'entraînement
existing_cols_to_drop_new = [col for col in cols_to_drop if col in data_new.columns]
X_new = data_new.drop(columns=existing_cols_to_drop_new, errors='ignore')  # Caractéristiques

# Gérer les valeurs manquantes dans les nouvelles données
X_new = imputer.transform(X_new)

# Standardiser les nouvelles données
X_new = scaler.transform(X_new)

# Faire des prédictions sur les nouvelles données
new_predictions = model.predict(X_new)

# Ajouter les prédictions aux nouvelles données
data_new['predicted_bug_category'] = new_predictions

# Supprimer les colonnes contenant 'missing'
data_new = data_new.loc[:, (data_new != 'missing').any(axis=0)]

# Enregistrer les résultats dans un nouveau fichier Excel
output_path = 'machine_learning/Test_result_Annab.xlsx'
data_new.to_excel(output_path, index=False)

print(f"Predictions saved to {output_path}")
