import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

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

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Créer un autoencoder
input_dim = X_train.shape[1]
encoding_dim = 10  # Dimension de l'espace latent

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Entraîner l'autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Encoder les données d'entraînement et de test
encoder_model = Model(inputs=input_layer, outputs=encoder)
X_train_encoded = encoder_model.predict(X_train)
X_test_encoded = encoder_model.predict(X_test)

# Créer et entraîner un modèle MLP sur les caractéristiques encodées
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train_encoded, y_train)

# Prédire les étiquettes pour les données de test
y_pred = mlp.predict(X_test_encoded)

# Évaluer les performances du modèle
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Afficher les résultats
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Afficher le score de précision en pourcentage
print(f"\nAccuracy Score: {accuracy * 100:.2f}%")