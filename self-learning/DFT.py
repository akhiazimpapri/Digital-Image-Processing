import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg', 0)

# Apply DFT (convert to frequency domain)
dft = np.fft.fft2(img)
shifted_dft = np.fft.fftshift(dft)
magnitude_dft = 20 * np.log(np.abs(shifted_dft) + 1)

# Apply Inverse DFT (convert back to spatial domain)
inverse_shift = np.fft.ifftshift(shifted_dft)
reconstructed_img = np.fft.ifft2(inverse_shift)
reconstructed_img = np.abs(reconstructed_img)

# Calculate difference
difference = np.abs(img - reconstructed_img)
sum_difference = np.sum(difference)
avg_difference = np.mean(difference)

print("Total Sum of Differences:", sum_difference)
print("Average Difference per Pixel:", avg_difference)

# Show Original, DFT, Reconstructed, and Difference images
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(magnitude_dft, cmap='gray')
plt.title('DFT (Frequency Domain)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(reconstructed_img, cmap='gray')
plt.title('Reconstructed Image (Inverse DFT)')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(difference, cmap='gray')
plt.title('Difference Image')
plt.axis('off')

plt.show()
