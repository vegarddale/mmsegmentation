# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:00:14 2024

@author: vegar
"""

import os
from PIL import Image
import numpy as np


# =============================================================================
# compute class weights
# =============================================================================

label_dir = './data/shorelines/ann_dir'  # Directory containing the label files
class_counts = {}  # Dictionary to store counts of each class
total_pixels = 0  # Total number of pixels processed

# Step 1: Iterate through each label file
for img_root_path in os.listdir(label_dir):
    for img_path in os.listdir(f"{label_dir}/{img_root_path}"):
        print(img_path)
        if img_path.endswith(".png"):
            img = np.array(Image.open(os.path.join(f"{label_dir}/{img_root_path}", img_path)))
            # Update total pixels
            total_pixels += img.size
    
            # Update class counts
            for class_id in np.unique(img):
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += np.count_nonzero(img == class_id)

# Step 2: Compute class weights
num_classes = len(class_counts)
class_weights = {}
for class_id, count in class_counts.items():
    class_weights[class_id] = total_pixels / (num_classes * count)

print(class_weights)

weights_sorted = []
for i in range(0,9):
    if i in class_weights:
        weights_sorted.append(class_weights[i])
        
print(weights_sorted)
        
# =============================================================================
# indentify the class distributions
# =============================================================================

# label_dir = './data/shorelines/ann_dir'  # Directory containing the label files
# class_counts = {}  # Dictionary to store counts of each class
# total_pixels = 0  # Total number of pixels processed

# # Step 1: Iterate through each label file
# for img_root_path in os.listdir(label_dir):
#     for img_path in os.listdir(f"{label_dir}/{img_root_path}"):
#         print(img_path)
#         if img_path.endswith(".png"):
#             img = np.array(Image.open(os.path.join(f"{label_dir}/{img_root_path}", img_path)))
    
#             # Update total pixels
#             total_pixels += len(np.where(img.flatten()>0)[0])
    
#             # Update class counts
#             for class_id in range(1,8):
#                 if class_id not in class_counts:
#                     class_counts[class_id] = 0
#                 class_counts[class_id] += np.count_nonzero(img == class_id)

# # Step 2: Calculate and display the ratio for each class

# for class_id, count in class_counts.items():
#     class_ratio = (count / total_pixels) * 100
#     print(f"Class {class_id}: {class_ratio:.2f}% of the dataset")




