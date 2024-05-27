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

def extract_color_stats(image_dir, mask_dir):
    # Lire l'image et le masque
    image = cv2.imread(image_dir)
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    
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
def median_std(image, mask):
    bee_isolation = cv2.bitwise_and(image, image, mask=mask)
    bee_pixels = bee_isolation[mask != 0]
    bee_pixels = bee_pixels.reshape(-1, 3)
    median_values = np.median(bee_pixels, axis=0)
    std_values = np.std(bee_pixels, axis=0)
    result = np.concatenate((median_values, std_values))
    if len(result) != 6:
        return [None]*6
    return result.tolist()


#Process directory 
def process_directory(images_dir, masks_dir, output_file):
    results = []
    for image_filename in os.listdir(images_dir):
        if image_filename.endswith('.jpg'):  # Ensure this matches your image file extensions
            image_path = os.path.join(images_dir, image_filename)
            mask_filename = image_filename.replace('.jpg', '_mask.png')  # Adjust mask file pattern as necessary
            mask_path = os.path.join(masks_dir, mask_filename)
            
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is not None and mask is not None:
                stats = median_std(image, mask)
                results.append([image_filename] + stats)
            else:
                print(f"Failed to load image or mask for {image_filename}")

    # Convert results to a DataFrame and save as CSV
    df = pd.DataFrame(results, columns=['Image', 'Median Red', 'Median Green', 'Median Blue', 'Std Dev Red', 'Std Dev Green', 'Std Dev Blue'])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Define your directories and output file
images_dir = 'train\images_1_to_250'
masks_dir = 'train\masks'
output_file = 'train\classif.xlsx'

# Process the images
process_directory(images_dir, masks_dir, output_file)