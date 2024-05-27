import numpy as np
import os
import cv2
import pandas as pd
import datetime


starttime = datetime.datetime.now()


# Feature 1 - Symmetric index

def symmetric_index(image):
    #Convertir l'image en niveaux de gris
    gray_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Diviser l'image en deux parties
    half_width = gray_array.shape[1] // 2
    left_half = gray_array[:, :half_width]
    right_half = gray_array[:, half_width:]
    if left_half.shape[1] != right_half.shape[1]:
        right_half = cv2.resize(right_half, (half_width, gray_array.shape[0]))
    
    #Calculer la symétrie
    symmetry = np.sum(np.abs(left_half - np.flip(right_half, axis=1))) / np.prod(left_half.shape)
    return symmetry


# Feature 2 - The ratio between the 2 longest orthogonal lines that can cross the bug

# Feature 3 - The ratio of the number of pixels of bug divided by the number of pixels of the full image

# Feature 4 - The min, max and mean values for Red, Green and Blue within the bug mask

def extract_color_stats(image_folder, mask_folder):
    # Lire l'image et le masque
    image = cv2.imread(image_folder)
    mask = cv2.imread(mask_folder, cv2.IMREAD_GRAYSCALE)
    
    # Appliquer le masque pour extraire les pixels de l'insect
    bug_pixels = cv2.bitwise_and(image, image, mask=mask)
    
    # Séparer les canaux de couleur
    blue_channel = bug_pixels[:, :, 0]
    green_channel = bug_pixels[:, :, 1]
    red_channel = bug_pixels[:, :, 2]
    
    # Filtrer les pixels non-masqués
    blue_values = blue_channel[mask > 0]
    green_values = green_channel[mask > 0]
    red_values = red_channel[mask > 0]
    
    # Calculer les statistiques
    stats = {
        "Red": {
            "min": np.min(red_values),
            "max": np.max(red_values),
            "mean": np.mean(red_values)
        },
        "Green": {
            "min": np.min(green_values),
            "max": np.max(green_values),
            "mean": np.mean(green_values)
        },
        "Blue": {
            "min": np.min(blue_values),
            "max": np.max(blue_values),
            "mean": np.mean(blue_values)
        }
    }
    
    return stats


# Feature 5 - The median and standard deviation for the Red, Green and Blue within the bug mask