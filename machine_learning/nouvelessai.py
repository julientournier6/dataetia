import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
import xgboost as xgb
import pickle

# Feature extraction functions
def calculate_symmetry_index(image):
    if len(image.shape) == 3:
        image = image[:, :, 0]
    height, width = image.shape
    left_half = image[:, :width // 2]
    right_half = image[:, width // 2:]
    right_half_flipped = np.fliplr(right_half)
    left_half = left_half.astype(np.float32)
    right_half_flipped = right_half_flipped.astype(np.float32)
    diff = cv2.absdiff(left_half, right_half_flipped)
    score = np.sum(diff) / (height * (width // 2))
    return score

def calculate_orthogonal_ratio(mask):
    points = np.column_stack(np.where(mask > 0))
    if points.shape[0] < 5:
        return 0
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    distances = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
    sorted_distances = sorted(distances)
    orthogonal_ratio = sorted_distances[0] / sorted_distances[2]
    return orthogonal_ratio

def calculate_bug_pixel_ratio(mask):
    total_pixels = mask.size
    bug_pixels = np.sum(mask > 0)
    bug_pixel_ratio = bug_pixels / total_pixels
    return bug_pixel_ratio

def calculate_color_statistics(image, mask):
    mask = (mask * 255).astype(np.uint8)
    bug_isolation = cv2.bitwise_and(image, image, mask=mask)
    bug_pixels = bug_isolation[mask != 0]
    if bug_pixels.size == 0:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
    bug_pixels = bug_pixels.reshape(-1, 3)
    min_values = np.min(bug_pixels, axis=0)
    max_values = np.max(bug_pixels, axis=0)
    mean_values = np.mean(bug_pixels, axis=0)
    median_values = np.median(bug_pixels, axis=0)
    std_values = np.std(bug_pixels, axis=0)
    return min_values, max_values, mean_values, median_values, std_values

def calculate_additional_features(mask):
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        return perimeter, area
    return 0, 0

def load_images_and_masks(image_dir, mask_dir, image_size=(128, 128)):
    images = []
    masks = []
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    for img_filename in image_filenames:
        img_id = os.path.splitext(img_filename)[0]
        mask_filename = f'binary_{img_id}.tif'
        
        if mask_filename in mask_filenames:
            img_path = os.path.join(image_dir, img_filename)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, image_size)
            image = image / 255.0
            images.append(image)
            
            mask_path = os.path.join(mask_dir, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = cv2.resize(mask, image_size)
            mask = mask / 255.0
            masks.append(mask)
    
    return np.array(images), np.array(masks)

def extract_features(images, masks):
    features = []
    for i, (image, mask) in enumerate(zip(images, masks)):
        if image is None or mask is None:
            continue
        
        min_values, max_values, mean_values, median_values, std_values = calculate_color_statistics(image, mask)
        perimeter, area = calculate_additional_features(mask)
        pixel_ratio = calculate_bug_pixel_ratio(mask)
        symmetry_index = calculate_symmetry_index(image)
        orthogonal_ratio = calculate_orthogonal_ratio(mask)
        
        features.append([
            min_values[2], min_values[1], min_values[0],
            max_values[2], max_values[1], max_values[0],
            mean_values[2], mean_values[1], mean_values[0],
            median_values[2], median_values[1], median_values[0],
            std_values[2], std_values[1], std_values[0],
            perimeter, area, pixel_ratio, symmetry_index, orthogonal_ratio
        ])
        
    return np.array(features)

# Load previous data if it exists
features_file = 'features.npy'
labels_file = 'labels.npy'

if os.path.exists(features_file) and os.path.exists(labels_file):
    features = np.load(features_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)
else:
    features = np.empty((0, 20))
    labels = np.empty(0)

# Load new images and masks
image_dir = 'train/images_250_to_347'
mask_dir = 'train/masks_251_to_347'
images, masks = load_images_and_masks(image_dir, mask_dir)

# Check for the number of images and masks
print(f"Number of images: {len(images)}")
print(f"Number of masks: {len(masks)}")

# Extract new features
new_features = extract_features(images, masks)

# Check if features were extracted successfully
if new_features.size == 0:
    raise ValueError("No features were extracted. Check your images and masks.")
else:
    print(f"Extracted features shape: {new_features.shape}")

# Load labels
new_labels = pd.read_excel('Machine_learning/données.xlsx')['bug type'].values

# Adjust the labels to match the filtered images and masks
new_labels = new_labels[:len(new_features)]

# Combine with previous data
features = np.vstack((features, new_features))
labels = np.concatenate((labels, new_labels))

# Save the updated features and labels
np.save(features_file, features, allow_pickle=True)
np.save(labels_file, labels, allow_pickle=True)

# Create a dictionary to count the frequency of each class
label_counts = pd.Series(labels).value_counts()
total_samples = len(labels)
threshold = 0.05 * total_samples  # 5% threshold

# Map less frequent classes to "other"
labels = np.array([label if label_counts[label] >= threshold else 'other' for label in labels])

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Ensure all classes are present
all_classes = np.arange(labels_encoded.max() + 1)
unique_classes = np.unique(labels_encoded)
missing_classes = np.setdiff1d(all_classes, unique_classes)

# Add dummy samples for missing classes
for cls in missing_classes:
    dummy_feature = np.zeros((1, features.shape[1]))
    features = np.vstack([features, dummy_feature])
    labels_encoded = np.append(labels_encoded, cls)

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Define the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Setup the randomized search with cross-validation
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, cv=StratifiedKFold(n_splits=5), n_jobs=-1, scoring='accuracy')

# Fit the model with randomized search
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the final model with the best parameters
best_model = random_search.best_estimator_

# Save the trained model and the label encoder
with open('model.pkl', 'wb') as file:
    pickle.dump((best_model, le), file)

print("Model and LabelEncoder trained and saved as model.pkl with best parameters")
