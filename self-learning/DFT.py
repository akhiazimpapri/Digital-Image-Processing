import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg', 0)

# Apply DFT using NumPy
dft = np.fft.fft2(img)
shifted_dft = np.fft.fftshift(dft)
magnitude_dft = 20 * np.log(np.abs(shifted_dft) + 1)

# Show original and DFT result
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_dft, cmap='gray')
plt.title('DFT (Frequency Domain)')
plt.axis('off')

plt.show()
