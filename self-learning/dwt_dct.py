import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Load Image (Grayscale)
img = cv2.imread('/Users/akhi/Desktop/DIP/images/HVD.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found. Check the file path.")

# Discrete Cosine Transform (DCT)
    # Convert to float32 for DCT
img_float = np.float32(img) / 255.0

# Apply DCT
dct = cv2.dct(img_float)

# Use log transform for visualization (enhances frequency contrast)
dct_log = np.log(abs(dct) + 1)


# Discrete Wavelet Transform (DWT)
    # Apply single-level 2D DWT using Haar wavelet
coeffs2 = pywt.dwt2(img, 'haar')
LL, (LH, HL, HH) = coeffs2

# Reconstruct image from DWT coefficients
reconstructed_dwt = pywt.idwt2(coeffs2, 'haar')

# Visualization
plt.figure(figsize=(8,8))

# Original
plt.subplot(3,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# DCT
plt.subplot(3,3,2)
plt.imshow(dct_log, cmap='gray')
plt.title('DCT (Frequency Domain)')
plt.axis('off')

# DWT
plt.subplot(3,3,4)
plt.imshow(LL, cmap='gray')
plt.title('LL (Approximation)')
plt.axis('off')

plt.subplot(3,3,5)
plt.imshow(LH, cmap='gray')
plt.title('LH (Horizontal Detail)')
plt.axis('off')

plt.subplot(3,3,6)
plt.imshow(HL, cmap='gray')
plt.title('HL (Vertical Detail)')
plt.axis('off')

plt.subplot(3,3,7)
plt.imshow(HH, cmap='gray')
plt.title('HH (Diagonal Detail)')
plt.axis('off')

plt.subplot(3,3,8)
plt.imshow(reconstructed_dwt, cmap='gray')
plt.title('Reconstructed Image (IDWT)')
plt.axis('off')

plt.tight_layout()
plt.show()
