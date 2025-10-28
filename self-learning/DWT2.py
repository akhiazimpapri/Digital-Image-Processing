import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# ------------------------------
# Step 1: Load grayscale image
# ------------------------------
img = cv2.imread('/Users/akhi/Desktop/DIP/images/HVD.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found! Check the file path.")

# ------------------------------
# Step 2: Apply 2D Discrete Wavelet Transform
# ------------------------------
LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')

# ------------------------------
# Step 3: Compute threshold based on LL
# ------------------------------
t = np.max(LL) * 0.2   # 20% of max(LL) as threshold

# ------------------------------
# Step 4: Apply hard thresholding to LH
# ------------------------------
LH_thresh = np.where(np.abs(LH) < t, 0, LH)

# ------------------------------
# Step 5: Reconstruct image from modified coefficients
# ------------------------------
coeffs2_thresh = (LL, (LH_thresh, HL, HH))
img_reconstructed = pywt.idwt2(coeffs2_thresh, 'haar')

# ------------------------------
# Step 6: Display subbands and results
# ------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(LL, cmap='gray')
plt.title('LL (Approximation)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(LH, cmap='gray')
plt.title('LH (Original Horizontal Detail)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(LH_thresh, cmap='gray')
plt.title('LH_thresh (Thresholded LH)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(HL, cmap='gray')
plt.title('HL (Vertical Detail)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(HH, cmap='gray')
plt.title('HH (Diagonal Detail)')
plt.axis('off')

plt.tight_layout()
plt.show()

# ------------------------------
# Step 7: Display reconstructed image separately
# ------------------------------
plt.figure(figsize=(5, 5))
plt.imshow(img_reconstructed, cmap='gray')
plt.title('Reconstructed Image (From coeffs2_thresh)')
plt.axis('off')
plt.show()

# ------------------------------
# Step 8: Save reconstructed image
# ------------------------------
save_path = '/Users/akhi/Desktop/DIP/images/HVD_reconstructed.png'
cv2.imwrite(save_path, img_reconstructed)

print(f"✅ Reconstructed image saved successfully at: {save_path}")
