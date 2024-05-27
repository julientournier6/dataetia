import os
import numpy as np
from PIL import Image
import pandas as pd
from os.path import exists

def calculate_bug_pixel_ratio(image_path):
    try:
        # Charger l'image
        image = Image.open(image_path).convert('RGBA')
        # Convertir en Numpy array
        data = np.array(image)
        
        # Calculer le nombre total de pixels
        total_pixels = data.shape[0] * data.shape[1]
        
        # Calculer le nombre de pixels non transparents
        non_transparent_pixels = np.sum(data[:, :, 3] > 0)
        
        # Calculer le ratio
        bug_pixel_ratio = non_transparent_pixels / total_pixels
        
        print(f"Image: {image_path}, Bug pixel ratio: {bug_pixel_ratio}")
        return bug_pixel_ratio
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {image_path}: {e}")
        return np.nan

def process_images_in_folder(folder_path):
    try:
        # Préparer le DataFrame
        df = pd.read_csv('classif.csv')
        print("CSV chargé avec succès")

        df["bug_pixel_ratio"] = np.nan  # Ratio du nombre de pixels de l'insecte par rapport au nombre total de pixels
        
        # Parcourir les images dans le dossier
        for id in df["ID"]:
            filename = str(id) + '.png'
            image_path = os.path.join(folder_path, filename)
            print(f"Chemin de l'image: {image_path}")
            
            if exists(image_path):
                # Calculer le ratio du nombre de pixels de l'insecte
                bug_pixel_ratio = calculate_bug_pixel_ratio(image_path)
                
                # Ajouter les résultats au DataFrame
                df.loc[df['ID'] == id, ['bug_pixel_ratio']] = [bug_pixel_ratio]
                print(f"Valeur ajoutée pour l'ID {id}: {bug_pixel_ratio}")
        
        # Sauvegarder les résultats dans un fichier CSV
        df.to_csv('classif.csv', index=False)
        print("CSV sauvegardé avec succès")
    except Exception as e:
        print(f"Erreur lors du traitement des images dans le dossier {folder_path}: {e}")

# Remplacez 'path_to_your_folder' par le chemin de votre dossier contenant les images
folder_path = 'C:/DATAIA/train/images_1_to_250'
process_images_in_folder(folder_path)
