import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Charger les données
data_train = pd.read_excel('machine_learning/données.xlsx')

# Normaliser les noms de colonnes en supprimant les suffixes _x et _y et en les convertissant en minuscules
data_train.columns = [col.split('_')[0].strip().lower() for col in data_train.columns]

# Supprimer les colonnes inutiles pour l'entraînement
X = data_train.drop(columns=['id', 'bug type', 'species'])
y = data_train['bug type']

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construire l'autoencodeur
input_dim = X_train.shape[1]
encoding_dim = 32  # Vous pouvez ajuster ce nombre de dimensions encodées

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Entraîner l'autoencodeur
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)

# Utiliser l'encodeur pour obtenir les caractéristiques réduites
encoder_model = Model(inputs=input_layer, outputs=encoder)
X_train_encoded = encoder_model.predict(X_train)
X_test_encoded = encoder_model.predict(X_test)

# Appliquer un modèle supervisé sur les caractéristiques encodées
# Choix 1: Régression Logistique
logistic_regression = LogisticRegression(random_state=42, max_iter=200)
logistic_regression.fit(X_train_encoded, y_train)
y_pred_lr = logistic_regression.predict(X_test_encoded)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy: {:.2f}%".format(accuracy_lr * 100))

# Choix 2: Perceptron Multicouche (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train_encoded, y_train)
y_pred_mlp = mlp.predict(X_test_encoded)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLP Accuracy: {:.2f}%".format(accuracy_mlp * 100))