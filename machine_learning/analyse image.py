import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
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
                print(f"Failed to load image: {img_path}")
                continue
            image = cv2.resize(image, image_size)
            image = image / 255.0
            images.append(image)
            
            mask_path = os.path.join(mask_dir, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                continue
            mask = cv2.resize(mask, image_size)
            mask = mask / 255.0
            masks.append(mask)
        else:
            print(f"No corresponding mask found for image: {img_filename}")
    
    return np.array(images), np.array(masks)

def extract_features(images, masks):
    features = []
    for i, (image, mask) in enumerate(zip(images, masks)):
        print(f"Processing image and mask {i+1}/{len(images)}")
        if image is None or mask is None:
            print(f"Image or mask {i+1} is None")
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

# Load the trained model and LabelEncoder
with open('model.pkl', 'rb') as file:
    model, le = pickle.load(file)

# Load and process new images and masks
image_dir = 'train/images_1_to_250'
mask_dir = 'train/masks_1_to_250'
images, masks = load_images_and_masks(image_dir, mask_dir)

# Extract features
features = extract_features(images, masks)

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Predict using the trained model
predictions = model.predict(features_scaled)
predicted_labels = le.inverse_transform(predictions)

# Load actual insect types from données.xlsx for comparison
actual_data = pd.read_excel('Machine learning/données.xlsx')
actual_labels = actual_data['bug type'].values
actual_image_ids = actual_data['ID'].astype(str).values

# Map species to bee, bumblebee, or other
mapped_actual_labels = np.array(['bee' if 'bee' in label.lower() else 'bumblebee' if 'bumblebee' in label.lower() else 'other' for label in actual_labels])

# Initialize counters for accuracy calculation
correct_predictions = 0
total_predictions = len(predicted_labels)

# Compare predictions with actual insect types
for i, pred in enumerate(predicted_labels):
    image_id = os.path.splitext(os.listdir(image_dir)[i])[0]
    actual_index = np.where(actual_image_ids == image_id)[0]
    if len(actual_index) > 0:
        actual_label = mapped_actual_labels[actual_index[0]]
        if actual_label == pred:
            correct_predictions += 1
        print(f"Image {i+1}: Actual insect type: {actual_label}, Predicted insect type: {pred}")
    else:
        print(f"Image {i+1}: Actual insect type: Not found, Predicted insect type: {pred}")

# Calculate and print the percentage of correct predictions
accuracy_percentage = (correct_predictions / total_predictions) * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")
