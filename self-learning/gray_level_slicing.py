import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = plt.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg')

if img.max() <= 1:
    img = (img * 255).astype(np.uint8)

# Define slicing range
r1, r2 = 100, 150

# Gray-level slicing with background
sliced = img.copy()
sliced[(img >= r1) & (img <= r2)] = 255

# Display
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(sliced, cmap='gray')
plt.title("Gray-Level Slicing")
plt.axis('off')

plt.show()
