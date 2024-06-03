import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from keras.layers import Input, Dense
from keras.models import Model
import logging

logging.basicConfig(level=logging.DEBUG)

# Chemin relatif pour le fichier de données
file_path = 'machine_learning/données.xlsx'

# Charger les données depuis le fichier Excel
logging.info("Loading data from Excel file")
data = pd.read_excel(file_path)

# Afficher un aperçu des données
logging.debug(f"Data head:\n{data.head()}")
logging.debug(f"Data types:\n{data.dtypes}")

# Séparer les caractéristiques (features) et les étiquettes (labels)
X = data.drop(columns=['ID', 'bug type', 'species'])
y = data['bug type']

# Diviser les données en ensembles d'entraînement et de test
logging.info("Splitting data into train and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les données
logging.info("Standardizing the data")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construire l'autoencoder
logging.info("Building the autoencoder")
input_dim = X_train_scaled.shape[1]
encoding_dim = 10

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner l'autoencoder
logging.info("Training the autoencoder")
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=16, shuffle=True, validation_split=0.2)

# Encoder les caractéristiques
logging.info("Encoding the features using the trained autoencoder")
encoder = Model(input_layer, encoded)
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

# Entraîner un MLP sur les caractéristiques encodées
logging.info("Training MLP on the encoded features")
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train_encoded, y_train)

# Prédire et évaluer le modèle
logging.info("Predicting and evaluating the model")
y_pred = mlp.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Accuracy: {accuracy * 100:.2f}%")