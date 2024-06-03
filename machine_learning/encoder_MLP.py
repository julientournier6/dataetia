import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Charger les données
data_train = pd.read_excel('machine_learning/données.xlsx')

# Normaliser les noms de colonnes en supprimant les suffixes _x et _y et en les convertissant en minuscules
data_train.columns = [col.split('_')[0].strip().lower() for col in data_train.columns]

# Supprimer les colonnes inutiles pour l'entraînement
X_train = data_train.drop(columns=['id', 'bug type', 'species'])

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Définir l'architecture de l'autoencodeur
input_dim = X_train_scaled.shape[1]
encoding_dim = 32  # Vous pouvez ajuster cette valeur

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Compiler l'autoencodeur
autoencoder.compile(optimizer='adam', loss='mse')

# Entraîner l'autoencodeur
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=100, batch_size=16, shuffle=True, validation_split=0.2)

# Encoder les caractéristiques du dataset d'entraînement
X_train_encoded = encoder.predict(X_train_scaled)

# Sauvegarder les caractéristiques encodées dans un fichier Excel
encoded_df = pd.DataFrame(X_train_encoded)
encoded_df.to_excel('machine_learning/encoded_données.xlsx', index=False)

print("Caractéristiques encodées enregistrées dans 'machine_learning/encoded_données.xlsx'")