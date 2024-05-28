import os
import numpy as np
import cv2
import pandas as pd

images_dir = 'train/images_1_to_250'
masks_dir = 'train/masks'
output_file = 'train/testratio.xlsx'

def calculate_bug_pixel_ratio(mask):
    # Calculer le nombre total de pixels
    total_pixels = mask.size
    
    # Calculer le nombre de pixels non noirs (pixels de l'insecte)
    bug_pixels = np.sum(mask > 0)
    
    # Calculer le ratio
    bug_pixel_ratio = bug_pixels / total_pixels
    
    return bug_pixel_ratio

def process_directory(images_dir, masks_dir, output_file):
    results = []
    
    for image_filename in os.listdir(images_dir):
        if image_filename.lower().endswith('.jpg'):  # Assurez-vous que cela correspond aux extensions de fichiers image
            image_id = os.path.splitext(image_filename)[0]
            image_path = os.path.join(images_dir, image_filename)
            mask_filename = f'binary_{image_id}.tif'  # Ajuster le modèle de fichier de masque si nécessaire
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
            bug_pixel_ratio = calculate_bug_pixel_ratio(mask)
            results.append({'Filename': image_filename, 'Bug Pixel Ratio': bug_pixel_ratio})
            print(f"Image: {image_filename}, Bug Pixel Ratio: {bug_pixel_ratio}")

    # Convertir les résultats en DataFrame
    if results:
        df = pd.DataFrame(results)
        print(df.head())  # Afficher les premières lignes du DataFrame pour vérification
        
        # Sauvegarder les résultats dans un fichier Excel
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")

# Process the images and save the results
process_directory(images_dir, masks_dir, output_file)
