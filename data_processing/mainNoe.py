import numpy as np
import cv2
import pandas as pd
import os

# Fonction pour calculer l'indice de symétrie basé sur les contours de l'insecte
def calculate_symmetry_index(contour):
    if len(contour) < 5:
        return 0
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    distances = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
    sorted_distances = sorted(distances)
    symmetry_index = sorted_distances[1] / sorted_distances[3]
    return symmetry_index

# Fonction pour calculer le ratio des deux lignes orthogonales les plus longues
def calculate_orthogonal_ratio(contour):
    if len(contour) < 5:
        return 0
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    distances = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
    sorted_distances = sorted(distances)
    orthogonal_ratio = sorted_distances[0] / sorted_distances[2]  # Smallest divided by longest
    return orthogonal_ratio

# Fonction pour calculer le ratio de pixels de l'insecte par rapport aux pixels totaux de l'image
def calculate_bug_pixel_ratio(mask):
    total_pixels = mask.size
    bug_pixels = np.sum(mask > 0)
    bug_pixel_ratio = bug_pixels / total_pixels
    return bug_pixel_ratio

# Fonction pour calculer les statistiques des couleurs (min, max, moyenne, médiane, écart-type)
def calculate_color_statistics(image, mask):
    bee_isolation = cv2.bitwise_and(image, image, mask=mask)
    bee_pixels = bee_isolation[mask != 0]
    bee_pixels = bee_pixels.reshape(-1, 3)
    min_values = np.min(bee_pixels, axis=0)
    max_values = np.max(bee_pixels, axis=0)
    mean_values = np.mean(bee_pixels, axis=0)
    median_values = np.median(bee_pixels, axis=0)
    std_values = np.std(bee_pixels, axis=0)
    return min_values, max_values, mean_values, median_values, std_values

# Fonction pour calculer des fonctionnalités additionnelles (périmètre et aire)
def calculate_additional_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        return perimeter, area
    return 0, 0

# Fonction principale pour traiter une seule image et un masque
def process_image(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return None

    # Calculer les statistiques de couleur
    min_values, max_values, mean_values, median_values, std_values = calculate_color_statistics(image, mask)
    
    # Calculer des fonctionnalités additionnelles
    perimeter, area = calculate_additional_features(mask)
    
    # Calculer le ratio de pixels de l'insecte
    pixel_ratio = calculate_bug_pixel_ratio(mask)

    # Trouver les contours et calculer l'indice de symétrie
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symmetry_index = calculate_symmetry_index(contours[0]) if contours else 0

    # Calculer le ratio orthogonal
    orthogonal_ratio = calculate_orthogonal_ratio(contours[0]) if contours else 0

    return {
        "min_red": min_values[2], "min_green": min_values[1], "min_blue": min_values[0],
        "max_red": max_values[2], "max_green": max_values[1], "max_blue": max_values[0],
        "mean_red": mean_values[2], "mean_green": mean_values[1], "mean_blue": mean_values[0],
        "median_red": median_values[2], "median_green": median_values[1], "median_blue": median_values[0],
        "std_red": std_values[2], "std_green": std_values[1], "std_blue": std_values[0],
        "perimeter": perimeter, "area": area, "pixel_ratio": pixel_ratio, "symmetry_index": symmetry_index,
        "orthogonal_ratio": orthogonal_ratio
    }

# Fonction pour traiter toutes les images dans un répertoire donné
def process_directory(images_dir, masks_dir, output_file_excel, output_file_csv):
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

    # Convertir les résultats en DataFrame et enregistrer en fichier Excel et CSV
    columns = [
        'Image', 'Min Red', 'Min Green', 'Min Blue', 'Max Red', 'Max Green', 'Max Blue',
        'Mean Red', 'Mean Green', 'Mean Blue', 'Median Red', 'Median Green', 'Median Blue',
        'Std Dev Red', 'Std Dev Green', 'Std Dev Blue', 'Perimeter', 'Area', 'Pixel Ratio',
        'Symmetry Index', 'Orthogonal Ratio'
    ]
    if results:
        df = pd.DataFrame(results, columns=columns)
        print(df.head())  # Afficher les premières lignes du DataFrame pour vérification
        
        # Sauvegarder les résultats dans un fichier Excel
        df.to_excel(output_file_excel, index=False)
        print(f"Results saved to {output_file_excel}")
        
        # Sauvegarder les résultats dans un fichier CSV
        df.to_csv(output_file_csv, index=False)
        print(f"Results also saved to {output_file_csv}")
    else:
        print("No results to save.")

# Définir les répertoires et les fichiers de sortie
images_dir = 'train/images_1_to_250'
masks_dir = 'train/masks'
output_file_excel = 'train/classif.xlsx'
output_file_csv = 'train/classif.csv'

# Traiter les images et sauvegarder les résultats
process_directory(images_dir, masks_dir, output_file_excel, output_file_csv)