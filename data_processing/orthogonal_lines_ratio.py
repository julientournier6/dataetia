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