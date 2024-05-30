import cv2
import numpy as np
import pandas as pd
import os
### Feature 4 - Ratio between the 2 longest orthogonal lines that can cross the bug ###

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