import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg', cv2.IMREAD_GRAYSCALE)

# Normalize image to [0,1] for computation
img_norm = img / 255.0

# ---------------- Power-law (Gamma) transformation ----------------
gamma = 2.0   # you can try values like 0.5, 1.5, 2.0
c_gamma = 1.0
gamma_transformed = c_gamma * np.power(img_norm, gamma)
gamma_result = np.uint8(255 * gamma_transformed)

# ---------------- Logarithmic transformation ----------------
c_log = 255 / np.log2(1 + np.max(img))   # scaling constant
log_transformed = c_log * np.log2(1 + img.astype(np.float32))
log_result = np.uint8(log_transformed)

# ---------------- Display Images ----------------
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title(f"Gamma Transform (γ={gamma})")
plt.imshow(gamma_result, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Log Transform")
plt.imshow(log_result, cmap='gray')
plt.axis("off")

# ---------------- Display Histograms ----------------
plt.subplot(2, 3, 4)
plt.title("Histogram: Original")
plt.hist(img.ravel(), bins=256, range=(0, 256), color='black')

plt.subplot(2, 3, 5)
plt.title("Histogram: Gamma")
plt.hist(gamma_result.ravel(), bins=256, range=(0, 256), color='black')

plt.subplot(2, 3, 6)
plt.title("Histogram: Log")
plt.hist(log_result.ravel(), bins=256, range=(0, 256), color='black')

plt.tight_layout()
plt.show()
