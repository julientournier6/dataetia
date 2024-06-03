import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Charger les données
data_train = pd.read_excel('machine_learning/données.xlsx')
data_test = pd.read_excel('machine_learning/données2.xlsx')

# Supprimer les colonnes inutiles pour l'entraînement
X_train = data_train.drop(columns=['ID', 'bug type', 'species'])
y_train = data_train['bug type']

X_test = data_test.drop(columns=['ID'])

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construire l'auto-encodeur
input_dim = X_train_scaled.shape[1]
encoding_dim = 32  # Vous pouvez ajuster cette valeur

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Entraîner l'auto-encodeur
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Extraire les caractéristiques encodées
encoder_model = Model(inputs=input_layer, outputs=encoder)
X_train_encoded = encoder_model.predict(X_train_scaled)
X_test_encoded = encoder_model.predict(X_test_scaled)

# Entraîner un modèle supervisé sur les caractéristiques encodées
# Utiliser un MLP (Multi-layer Perceptron) pour cette tâche
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp.fit(X_train_encoded, y_train)

# Prédire les labels pour le nouveau dataset
y_test_pred = mlp.predict(X_test_encoded)

# Ajouter les prédictions au DataFrame de test
data_test['Predicted Bug Type'] = y_test_pred

# Enregistrer les prédictions dans un fichier Excel
data_test.to_excel('machine_learning/predicted_données2.xlsx', index=False)

print("Prédictions enregistrées dans 'machine_learning/predicted_données2.xlsx'")