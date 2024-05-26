import cv2
import numpy as np

def symmetry_index(image_path):
    # Lire l'image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Binariser l'image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
 
    # Trouver les contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)  # Prendre le plus grand contour
 
    # Créer un masque de l'objet
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
 
    # Calculer le centre de masse de l'objet
    M = cv2.moments(mask)
    if M["m00"] == 0:  # Éviter la division par zéro
        return float('inf')
    cx = int(M["m10"] / M["m00"])
 
    # Calculer les distances de symétrie
    h, w = mask.shape
    total_distance = 0
    count = 0
 
    for y in range(h):
        for x in range(cx):
            if mask[y, x] == 255:
                symmetric_x = cx + (cx - x)
                if symmetric_x < w and mask[y, symmetric_x] == 255:
                    total_distance += abs(mask[y, x] - mask[y, symmetric_x])
                    count += 1
 
    if count == 0:  # Éviter la division par zéro
        return float('inf')
   
    # Calculer l'indice de symétrie
    symmetry_index = total_distance / count
    return symmetry_index