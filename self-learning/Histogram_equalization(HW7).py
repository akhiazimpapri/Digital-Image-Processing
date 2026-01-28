import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Histogram Equalization Function
# --------------------------
def hist_equalization(img):
    # Flatten image into 1D array
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    
    # Normalize histogram (probability distribution)
    pdf = hist / hist.sum()
    
    # Compute cumulative distribution function (CDF)
    cdf = pdf.cumsum()
    #The CDF represents the cumulative sum of probabilities of gray levels up to a certain intensity value.
    # Normalize CDF to 0–255
    cdf_normalized = np.round(cdf * 255).astype(np.uint8)
    
    # Map original pixels to equalized values
    equalized_img = cdf_normalized[img]
    
    return equalized_img

# --------------------------
# Load Image
# --------------------------
img = cv2.imread("/Users/akhi/Desktop/DIP/images/tulip.png", cv2.IMREAD_GRAYSCALE)

# Apply own histogram equalization
my_equalized = hist_equalization(img)

# For comparison (OpenCV built-in, optional to check)
opencv_equalized = cv2.equalizeHist(img)

# --------------------------
# Plot Results
# --------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(my_equalized, cmap='gray')
plt.title("My Equalized Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(opencv_equalized, cmap='gray')
plt.title("OpenCV Equalized Image")
plt.axis("off")

# Histograms
plt.subplot(2, 3, 4)
plt.hist(img.flatten(), bins=256, color='black')
plt.title("Original Histogram")

plt.subplot(2, 3, 5)
plt.hist(my_equalized.flatten(), bins=256, color='black')
plt.title("My Equalized Histogram")

plt.subplot(2, 3, 6)
plt.hist(opencv_equalized.flatten(), bins=256, color='black')
plt.title("OpenCV Equalized Histogram")

plt.tight_layout()
plt.show()
