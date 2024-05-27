import cv2
import numpy as np
import os

def find_longest_orthogonal_lines(mask):
    # Find contours in the segmentation mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the largest contour, which should correspond to the insect
    if contours:
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
        
        return ratio, box
    else:
        return None, None

def process_images(image_folder, mask_folder, output_file):
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Image folder path: {image_folder}")
    print(f"Mask folder path: {mask_folder}")

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
            mask_file = image_file.replace('.jpg', '.tif').replace('.JPG', '.tif')
            mask_path = os.path.join(mask_folder, mask_file)
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing {image_file} with mask {mask_path}")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.imread(image_path)
                if mask is not None:
                    ratio, box = find_longest_orthogonal_lines(mask)
                    if ratio is not None:
                        results.append((image_file, ratio, box, image))
            else:
                print(f"Mask not found: {mask_path}")

    # Save the results to a CSV file
    if results:
        np.savetxt(output_file, [(res[0], res[1]) for res in results], fmt='%s,%.6f', delimiter=',', header='Image,Ratio', comments='')
        print(f'Feature extraction completed. Results saved to {output_file}')
    else:
        print("No valid results to save.")
    
    return results

if __name__ == "__main__":
    image_folder = 'train/images_1_to_250'
    mask_folder = 'train/masks'
    output_file = 'train/ratios.csv'
    
    results = process_images(image_folder, mask_folder, output_file)