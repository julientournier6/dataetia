import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Chemins des répertoires
test_images_dir = 'train/images_1_to_250'
test_masks_dir = 'train/masks'
classif_file = '/mnt/data/classif.xlsx'
output_file = 'Machine learning/testrun.xlsx'

# Fonction pour charger les images et les masques de test
def load_images_and_masks(images_dir, masks_dir):
    images = []
    masks = []
    filenames = sorted(os.listdir(images_dir))
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(images_dir, filename)
            mask_filename = f'binary_{image_id}.tif'
            mask_path = os.path.join(masks_dir, mask_filename)
            
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Failed to load image for {filename} at {image_path}")
            if mask is None:
                print(f"Failed to load mask for {filename} at {mask_path}")
            elif mask.sum() == 0:
                print(f"Mask for {filename} is empty.")
            
            if image is not None and mask is not None and mask.sum() > 0:
                images.append(image)
                masks.append(mask)
    return images, masks, filenames

# Fonction pour extraire les caractéristiques des images et des masques
def extract_features(images, masks):
    features = []
    for idx, (image, mask) in enumerate(zip(images, masks)):
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        masked_pixels = masked_image[mask != 0]
        
        if masked_pixels.size == 0:
            print(f"No masked pixels found for image {idx} with filename {filenames[idx]}, skipping.")
            continue
        
        min_values = np.min(masked_pixels, axis=0)
        max_values = np.max(masked_pixels, axis=0)
        mean_values = np.mean(masked_pixels, axis=0)
        median_values = np.median(masked_pixels, axis=0)
        std_values = np.std(masked_pixels, axis=0)
        
        feature = np.concatenate([min_values, max_values, mean_values, median_values, std_values])
        features.append(feature)
    return np.array(features)

# Charger les images et les masques de test
test_images, test_masks, filenames = load_images_and_masks(test_images_dir, test_masks_dir)

# Vérifier le nombre d'images et de masques chargés
print(f"Loaded {len(test_images)} images and {len(test_masks)} masks")

# Extraire les caractéristiques des images et des masques de test
test_features = extract_features(test_images, test_masks)

# Vérifier si les caractéristiques ont été correctement extraites
if test_features.size == 0:
    raise ValueError("Aucune caractéristique extraite. Vérifiez les images et les masques fournis.")
else:
    print(f"Extracted features shape: {test_features.shape}")

# Charger les données d'entraînement pour scaler et encoder
data_path = 'Machine learning/données.xlsx'
data = pd.read_excel(data_path)

# Préparer les données d'entraînement pour scaler et encoder
X = data.drop(columns=['ID', 'bug type'])
y = data['bug type']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X = X.select_dtypes(include=[float, int])
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_imputed)

# Imputer et scaler les caractéristiques de test
test_features_imputed = imputer.transform(test_features)
test_features_scaled = scaler.transform(test_features_imputed)

# Charger le modèle entraîné
model = Sequential([
    Dense(64, activation='relu', input_shape=(test_features_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('model.h5')  # Assurez-vous que le modèle est bien sauvegardé sous 'model.h5'

# Faire des prédictions
predictions = model.predict(test_features_scaled)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Charger le fichier classif.xlsx pour obtenir les identifiants d'image
df_classif = pd.read_excel(classif_file)

# Ajouter les prédictions au DataFrame
df_classif['Predicted Label'] = predicted_labels

# Enregistrer les résultats dans testrun.xlsx
df_classif.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")

# Afficher les résultats
for image, mask, filename, predicted_label in zip(test_images, test_masks, filenames, predicted_labels):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title(f'Predicted: {predicted_label}')
    
    plt.suptitle(f'Filename: {filename}')
    plt.show()
