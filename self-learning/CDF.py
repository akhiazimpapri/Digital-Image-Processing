import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image in grayscale
img = cv2.imread('/Users/akhi/Desktop/DIP/images/img.png', cv2.IMREAD_GRAYSCALE)

# 2. Calculate the histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist = hist.flatten()  # Convert to 1D array

# 3. Calculate PDF
pdf = hist / hist.sum()

# 4. Calculate CDF
cdf = np.cumsum(pdf)

# 5. Normalize CDF to 0-255
cdf_normalized = (cdf * 255).astype(np.uint8)

# 6. Plotting
plt.figure(figsize=(16,6))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2,3,2)
plt.plot(hist, color='black')
plt.title('Histogram')
plt.xlabel('Intensity Level')
plt.ylabel('Frequency')

plt.subplot(2,3,3)
plt.plot(pdf, color='blue')
plt.title('PDF')
plt.xlabel('Intensity Level')
plt.ylabel('Probability')

plt.subplot(2,3,4)
plt.plot(cdf, color='red')
plt.title('CDF')
plt.xlabel('Intensity Level')
plt.ylabel('Cumulative Probability')

plt.subplot(2,3,5)
plt.plot(cdf_normalized, color='green')
plt.title('Normalized CDF (0-255)')
plt.xlabel('Intensity Level')
plt.ylabel('Mapped Intensity')

plt.tight_layout()
plt.show()
