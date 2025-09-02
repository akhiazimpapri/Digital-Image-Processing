import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg')
img2 = cv2.imread('/Users/akhi/Desktop/DIP/images/roses.png')

img1 = cv2.resize(img1, (300,300))
img2 = cv2.resize(img2,(400,400))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

h1, w1 = gray1.shape
h2, w2 = gray2.shape


# Compute top-left corner to center small on large
height = (h2 - h1) // 2
width = (w2 - w1) // 2

gray3 = np.zeros((h2, w2), dtype=np.uint8)

# Copy small image to center of large canvas
for i in range(h1):
    for j in range(w1):
        gray3[height + i, width + j] = gray1[i, j]

# Display
plt.figure(figsize=(8, 8))

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title("Small Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.title("Large Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gray3, cmap='gray')
plt.title("Result (Small on Large)")
plt.axis("off")

plt.tight_layout()
plt.show()
