import cv2
import numpy as np
import pandas as pd
import os

def calculate_orthogonal_ratio(mask):
    points = np.column_stack(np.where(mask > 0))
    if points.shape[0] == 0:
        return None
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box) 
    
    # Calculer les longueurs des côtés du rectangle
    side_lengths = [np.linalg.norm(box[i] - box[(i+1) % 4]) for i in range(4)]
    side_lengths.sort(reverse=True)  # Trier les longueurs en ordre décroissant
    
    # Calculer le ratio entre les deux plus longues lignes orthogonales
    if side_lengths[3] != 0:
        ratio = side_lengths[2] / side_lengths[3]
        return ratio
    else:
        return None

def process_image(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return None

    # Calculer le ratio orthogonal
    orthogonal_ratio = calculate_orthogonal_ratio(mask)

    return {
        "orthogonal_ratio": orthogonal_ratio
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
            ortho_ratio = calculate_orthogonal_ratio(image)
            results.append({'Filename': image_filename, 'Bug Symmetry Index': ortho_ratio})
            print(f"Image: {image_filename}, Bug Symmetry Index: {ortho_ratio}")

    # Convertir les résultats en DataFrame
    if results:
        df = pd.DataFrame(results)
        print(df.head())  # Afficher les premières lignes du DataFrame pour vérification
        
        # Sauvegarder les résultats dans un fichier Excel
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")


# Définir les répertoires et le fichier de sortie
images_dir = 'train/images_1_to_250'
masks_dir = 'train/masks'
output_file = 'train/testratio.xlsx'

# Process the images and save the results
process_directory(images_dir, masks_dir, output_file)
    