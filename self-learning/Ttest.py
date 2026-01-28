import matplotlib.pyplot as plt
import numpy as np

# Load image (grayscale)
img = plt.imread('/Users/akhi/Desktop/DIP/images/images.jpeg')

# If image is RGB, convert to grayscale manually
if img.ndim == 3:
    img = img[:, :, 0]   # take one channel for simplicity

# If image is normalized (0–1), convert to 0–255
if img.max() <= 1.0:
    img = (img * 255).astype(np.uint8)

# Step 1: Initialize histogram array
hist = np.zeros(256, dtype=int)

# Step 2: Count pixel intensities manually
rows, cols = img.shape
for i in range(rows):
    for j in range(cols):
        intensity = img[i, j]
        hist[intensity] += 1

# Step 3: Plot histogram manually
plt.figure()
plt.plot(hist)
plt.title("Manual Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")
plt.show()
