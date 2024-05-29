import numpy as np
import os
import cv2
import pandas as pd
import datetime
import openpyxl

starttime = datetime.datetime.now()

# Feature 1 - Symmetric index

def symmetric_index(image):
    gray_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    half_width = gray_array.shape[1] // 2
    left_half = gray_array[:, :half_width]
    right_half = gray_array[:, half_width:]
    if left_half.shape[1] != right_half.shape[1]:
        right_half = cv2.resize(right_half, (half_width, gray_array.shape[0]))
    symmetry = np.sum(np.abs(left_half - np.flip(right_half, axis=1))) / np.prod(left_half.shape)
    return symmetry

# Feature 2 - The ratio between the 2 longest orthogonal lines that can cross the bug (smallest divided by longuest)

def orthogonal_lines_ratio(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    max_ratio = 0
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        side_lengths = [np.linalg.norm(box[i] - box[(i+1) % 4]) for i in range(4)]
        side_lengths.sort(reverse=True)
        if side_lengths[3] != 0:
            ratio = side_lengths[2] / side_lengths[3]
            if ratio > max_ratio:
                max_ratio = ratio
    return float(max_ratio)

# Feature 3 - The ratio of the number of pixels of bug divided by the number of pixels of the full image

def calculate_bug_pixel_ratio(mask):
    total_pixels = mask.size
    bug_pixels = np.sum(mask > 0)
    bug_pixel_ratio = bug_pixels / total_pixels
    return bug_pixel_ratio

# Feature 4 - The min, max and mean values for Red, Green and Blue within the bug mask

def extract_color_stats(image, mask):
    bug_pixels = cv2.bitwise_and(image, image, mask=mask)
    blue_channel = bug_pixels[:, :, 0]
    green_channel = bug_pixels[:, :, 1]
    red_channel = bug_pixels[:, :, 2]
    blue_values = blue_channel[mask > 0]
    green_values = green_channel[mask > 0]
    red_values = red_channel[mask > 0]
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

def process_directory(images_dir, masks_dir, output_excel, output_csv):
    results = []
    
    for image_filename in os.listdir(images_dir):
        if image_filename.lower().endswith('.jpg'):
            image_id = os.path.splitext(image_filename)[0]
            image_path = os.path.join(images_dir, image_filename)
            mask_filename = f'binary_{image_id}.tif'
            
            # Skip processing if the mask is 154
            if mask_filename != 'binary_154.tif':
                continue
            
            mask_path = os.path.join(masks_dir, mask_filename)
            
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is not None and mask is not None:
                symmetry = symmetric_index(image)
                ortho_ratio = orthogonal_lines_ratio(mask)
                bug_pixel_ratio = calculate_bug_pixel_ratio(mask)
                color_stats = extract_color_stats(image, mask)
                median_std_stats = median_std(image, mask)
                
                stats = [symmetry, ortho_ratio, bug_pixel_ratio,
                         color_stats['Red']['min'], color_stats['Red']['max'], color_stats['Red']['mean'],
                         color_stats['Green']['min'], color_stats['Green']['max'], color_stats['Green']['mean'],
                         color_stats['Blue']['min'], color_stats['Blue']['max'], color_stats['Blue']['mean']] + median_std_stats
                
                results.append([image_filename] + stats)
            else:
                print(f"Failed to load image or mask for {image_filename}")

    # Convert results to a DataFrame and save as Excel
    columns = ['Image', 'Symmetric Index', 'Ortho Ratio', 'Bug Pixel Ratio', 
               'Red Min', 'Red Max', 'Red Mean', 
               'Green Min', 'Green Max', 'Green Mean', 
               'Blue Min', 'Blue Max', 'Blue Mean', 
               'Median Red', 'Median Green', 'Median Blue', 
               'Std Dev Red', 'Std Dev Green', 'Std Dev Blue']
    
    df = pd.DataFrame(results, columns=columns)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")
    
    # Save the DataFrame as CSV
    df.to_csv(output_csv, index=False)
    print(f"Results also saved to {output_csv}")
    
    
#Execute in main 

if __name__ == "__main__":
    images_dir = 'train/images_1_to_250'
    masks_dir = 'train/masks'
    output_excel = 'train/classif.xlsx'
    output_csv = 'train/classif.csv'
    
    process_directory(images_dir, masks_dir, output_excel, output_csv)