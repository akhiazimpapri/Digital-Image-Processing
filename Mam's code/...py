import cv2
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the image
img_path = '/Users/akhi/Desktop/DIP/images/FLOWER.jpeg'
img = cv2.imread(img_path)

# Step 2: Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Find unique pixel values and their counts
u, c = np.unique(gray_img, return_counts=True)
print(u)
print(c)
