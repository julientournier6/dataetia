import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Charger les données
file_path = 'machine_learning/données.xlsx'
print(f"Loading data from {file_path}")
data = pd.read_excel(file_path)

# Préparer les caractéristiques et les labels
X = data.drop(columns=['ID', 'bug type', 'species'])
y = data['bug type']

# Diviser les données en ensembles d'entraînement et de test
print("Splitting data into train and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les caractéristiques
print("Standardizing the features")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définir et entraîner l'autoencodeur
input_dim = X_train_scaled.shape[1]
encoding_dim = 10

print("Building the autoencoder")
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

print("Training the autoencoder")
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test_scaled, X_test_scaled))

# Utiliser l'encodeur pour réduire la dimensionnalité des caractéristiques
print("Transforming the features using the encoder")
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

# Appliquer la régression logistique ou le MLP sur les caractéristiques encodées
print("Training the MLP classifier on the encoded features")
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train_encoded, y_train)

# Prédire les labels sur l'ensemble de test
print("Predicting the test set")
y_pred = mlp.predict(X_test_encoded)

# Calculer et afficher la précision
accuracy = np.mean(y_pred == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")