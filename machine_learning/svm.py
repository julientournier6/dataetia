import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Charger les données depuis le fichier Excel
file_path = 'machine_learning/données.xlsx'
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

# Afficher un aperçu des données
print(data.head())
print(data.dtypes)

# Séparer les caractéristiques (features) et l'étiquette (target)
X = data.drop(columns=['ID', 'bug type', 'species', 'bug_category'])  # Caractéristiques
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

# Créer le modèle SVM
model = SVC(kernel='linear', random_state=42)

# Entraîner le modèle
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

print(f"\nAccuracy Score: {accuracy * 100:.2f}%")