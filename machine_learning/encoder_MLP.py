import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Charger les données
data_train = pd.read_excel('classif_test_v2.xlsx')

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
encoding_dim = 128  # Vous pouvez ajuster ce nombre de dimensions encodées

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

# Entraîner l'autoencodeur
autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, shuffle=True, validation_split=0.2, verbose=1)

# Utiliser l'encodeur pour obtenir les caractéristiques réduites
encoder_model = Model(inputs=input_layer, outputs=encoder)
X_encoded = encoder_model.predict(X_scaled)
X_train_encoded = encoder_model.predict(X_train)
X_test_encoded = encoder_model.predict(X_test)

# Fonction pour évaluer un modèle avec cross-validation
def evaluate_model(model, X, y, cv_splits=5):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    return cv_scores.mean(), cv_scores.std()

# Appliquer un modèle supervisé sur les caractéristiques encodées avec validation croisée
# Choix 1: Régression Logistique
logistic_regression = LogisticRegression(random_state=42, max_iter=200)
mean_lr, std_lr = evaluate_model(logistic_regression, X_encoded, y)
print("Logistic Regression Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)".format(mean_lr * 100, std_lr * 100))

# Choix 2: Perceptron Multicouche (MLP) avec optimisations
mlp = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000, alpha=0.0001,
                    solver='adam', random_state=42, learning_rate_init=0.001)
mean_mlp, std_mlp = evaluate_model(mlp, X_encoded, y)
print("MLP Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)".format(mean_mlp * 100, std_mlp * 100))

# Entraîner le modèle MLP sur l'ensemble d'entraînement et évaluer sur l'ensemble de test pour comparaison
mlp.fit(X_train_encoded, y_train)
y_pred_mlp = mlp.predict(X_test_encoded)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLP Test Accuracy: {:.2f}%".format(accuracy_mlp * 100))
