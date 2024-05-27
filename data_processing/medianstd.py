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



image_path = 'train\images_1_to_250' 
mask_path = 'train\masks'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is not None and mask is not None:
    result = median_std(image, mask)
    print("Median and Std Dev RGB Values:", result)
else:
    print("Error loading image or mask")
    



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