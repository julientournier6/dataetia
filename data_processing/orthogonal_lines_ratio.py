import cv2
import numpy as np
import os

def find_longest_orthogonal_lines(mask):
    # Find contours in the segmentation mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the largest contour, which should correspond to the insect
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the longest orthogonal lines
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Calculate the lengths of the sides of the bounding box
    side_lengths = [np.linalg.norm(box[i] - box[(i+1) % 4]) for i in range(4)]
    side_lengths = sorted(side_lengths)
    
    # The ratio between the shortest and the longest of the two orthogonal lines
    ratio = side_lengths[0] / side_lengths[-1]
    
    return ratio

def process_images(image_folder, mask_folder, output_file):
    if not os.path.exists(image_folder):
        print(f"Error: Image folder {image_folder} does not exist.")
        return
    if not os.path.exists(mask_folder):
        print(f"Error: Mask folder {mask_folder} does not exist.")
        return

    image_files = os.listdir(image_folder)
    results = []

    for image_file in image_files:
        if image_file.lower().endswith('.jpg'):
            mask_file = image_file.replace('.JPG', '.tif').replace('.jpg', '.tif')  # Assuming masks are in .tif format
            mask_path = os.path.join(mask_folder, mask_file)
            
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    ratio = find_longest_orthogonal_lines(mask)
                    results.append((image_file, ratio))

    # Save the results to a CSV file
    np.savetxt(output_file, results, fmt='%s,%.6f', delimiter=',', header='Image,Ratio', comments='')

if __name__ == "__main__":
    image_folder = 'data_visualization/train/images_1_to_250'
    mask_folder = 'data_visualization/train/masks'
    output_file = 'data_visualization/train/ratios.csv'
    
    process_images(image_folder, mask_folder, output_file)