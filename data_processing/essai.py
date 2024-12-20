import numpy as np
import cv2
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops

# Feature 1 - Symmetry Index
def calculate_symmetry_index(image):
    # Si l'image a plus de 2 dimensions, garder uniquement la première (hauteur et largeur)
    if len(image.shape) == 3:
        image = image[:, :, 0]
    
    height, width = image.shape
    left_half = image[:, :width // 2]
    right_half = image[:, width // 2:]
    
    # Flip right half horizontally
    right_half_flipped = np.fliplr(right_half)
    
    # Convertir les deux matrices au même type de données
    left_half = left_half.astype(np.float32)
    right_half_flipped = right_half_flipped.astype(np.float32)
    
    # Calculate the symmetry index
    diff = cv2.absdiff(left_half, right_half_flipped)
    score = np.sum(diff) / (height * (width // 2))
    return score


# Feature 2 - The ratio between the 2 longest orthogonal lines that can cross the bug (smallest divided by longuest)
def calculate_orthogonal_ratio(mask):
    points = np.column_stack(np.where(mask > 0))
    if points.shape[0] < 5:  # Assurez-vous qu'il y a suffisamment de points pour calculer
        return 0
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    distances = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
    sorted_distances = sorted(distances)
    orthogonal_ratio = sorted_distances[0] / sorted_distances[2]  # Smallest divided by longest
    return orthogonal_ratio


# Feature 3 - The ratio of the number of pixels of bug divided by the number of pixels of the full image
def calculate_bug_pixel_ratio(mask):
    total_pixels = mask.size
    bug_pixels = np.sum(mask > 0)
    bug_pixel_ratio = bug_pixels / total_pixels
    return bug_pixel_ratio

# Feature 4/5 - The min, max and mean values for Red, Green and Blue within the bug mask. The median and standard deviation for the Red, Green and Blue within the bug mask
def calculate_color_statistics(image, mask):
    bug_isolation = cv2.bitwise_and(image, image, mask=mask)
    bug_pixels = bug_isolation[mask != 0]
    bug_pixels = bug_pixels.reshape(-1, 3)
    min_values = np.min(bug_pixels, axis=0)
    max_values = np.max(bug_pixels, axis=0)
    mean_values = np.mean(bug_pixels, axis=0)
    median_values = np.median(bug_pixels, axis=0)
    std_values = np.std(bug_pixels, axis=0)
    return min_values, max_values, mean_values, median_values, std_values

# Feature 6 - Excentricité (allongement de l'insecte) / Dans Symmetry_index (le temps que ça charge)
def calculate_eccentricity(mask):
    points = np.column_stack(np.where(mask > 0))
    if points.shape[0] >= 5:
        rect = cv2.minAreaRect(points)
        (center, axes, orientation) = rect
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
        if major_axis_length != 0:
            eccentricity = np.sqrt(1 - (minor_axis_length**2 / major_axis_length**2))
            return eccentricity
    return 0

#Feature 7 - Texture
def calculate_haralick_features(mask):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(mask, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    
    # Extraire les Haralick features
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        for i in range(glcm.shape[2]):
            feature = graycoprops(glcm, prop=prop)[0, i]
            features.append(feature)
    
    return features

#Feature 8 - Shape Descriptors
def calculate_shape_descriptors(mask):
    points = np.column_stack(np.where(mask > 0))
    if points.shape[0] >= 5:
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate area and perimeter directly from the mask
        area = np.sum(mask > 0)
        perimeter = cv2.arcLength(box, True)
        
        # Calculate circularity
        if perimeter != 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Calculate compactness
        if area != 0:
            compactness = perimeter ** 2 / area
        else:
            compactness = 0
        
        return circularity, compactness
    else:
        return 0, 0, 0

#Feature 9 - 


# Fonction principale pour traiter une seule image et un masque
def process_image(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return None

    # Calculer les statistiques de couleur
    min_values, max_values, mean_values, median_values, std_values = calculate_color_statistics(image, mask)
    
    # Calculer des fonctionnalités additionnelles
    eccentricity = calculate_eccentricity(mask)
    
    # Calculer le ratio de pixels de l'insecte
    pixel_ratio = calculate_bug_pixel_ratio(mask)

    # Calculer l'indice de symétrie
    symmetry_index = calculate_symmetry_index(image)

    # Calculer le ratio orthogonal
    orthogonal_ratio = calculate_orthogonal_ratio(mask)

    #Calculer les Haralick features
    haralick_features = calculate_haralick_features(mask)

    # Calculer les descripteurs de forme
    circularity, compactness = calculate_shape_descriptors(mask)
    

    return {
        "min_red": min_values[2], "min_green": min_values[1], "min_blue": min_values[0],
        "max_red": max_values[2], "max_green": max_values[1], "max_blue": max_values[0],
        "mean_red": mean_values[2], "mean_green": mean_values[1], "mean_blue": mean_values[0],
        "median_red": median_values[2], "median_green": median_values[1], "median_blue": median_values[0],
        "std_red": std_values[2], "std_green": std_values[1], "std_blue": std_values[0],
        "eccentricity":eccentricity , "pixel_ratio": pixel_ratio, "symmetry_index": symmetry_index,
        "orthogonal_ratio": orthogonal_ratio,
        "haralick_contrast": haralick_features[0], "haralick_dissimilarity": haralick_features[1],
        "haralick_homogeneity": haralick_features[2], "haralick_energy": haralick_features[3],
        "haralick_correlation": haralick_features[4], "haralick_ASM": haralick_features[5],
        "circularity": circularity, "compactness": compactness
    }

# Fonction pour traiter toutes les images dans un répertoire donné
def process_directory(images_dir, masks_dir, output_file):
    results = []
    
    for image_filename in os.listdir(images_dir):
        if image_filename.lower().endswith('.jpg'):
            image_id = os.path.splitext(image_filename)[0]
            image_path = os.path.join(images_dir, image_filename)
            mask_filename = f'binary_{image_id}.tif'
            mask_path = os.path.join(masks_dir, mask_filename)
            
            print(f"Attempting to load image: {image_path}")
            if not os.path.exists(image_path):
                print(f"Image file does not exist: {image_path}")
                continue
            
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            print(f"Attempting to load mask: {mask_path}")
            if not os.path.exists(mask_path):
                print(f"Mask file does not exist: {mask_path}")
                continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                continue
            
            print(f"Processing image: {image_filename}")
            stats = process_image(image_path, mask_path)
            if stats:
                results.append([image_filename] + list(stats.values()))
            else:
                print(f"Error processing image: {image_filename}")

    # Convertir les résultats en DataFrame et enregistrer en fichier Excel
    columns = [
        'Image', 'Min Red', 'Min Green', 'Min Blue', 'Max Red', 'Max Green', 'Max Blue',
        'Mean Red', 'Mean Green', 'Mean Blue', 'Median Red', 'Median Green', 'Median Blue',
        'Std Dev Red', 'Std Dev Green', 'Std Dev Blue', 'Eccentricity', 'Pixel Ratio',
        'Symmetry Index', 'Orthogonal Ratio','Haralick Contrast', 'Haralick Dissimilarity',
        'Haralick Homogeneity', 'Haralick Energy', 'Haralick Correlation', 'Haralick ASM',
        'Circularity', 'Compactness'
    ]
    if results:
        df = pd.DataFrame(results, columns=columns)
        print(df.head())  # Afficher les premières lignes du DataFrame pour vérification
        
        # Sauvegarder les résultats dans un fichier Excel
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")

# Définir les répertoires et le fichier de sortie
images_dir = 'train/images_1_to_250'
masks_dir = 'train/masks_1_to_250'
output_file = 'train/test_features.xlsx'


# Traiter les images et sauvegarder les résultats
process_directory(images_dir, masks_dir, output_file)