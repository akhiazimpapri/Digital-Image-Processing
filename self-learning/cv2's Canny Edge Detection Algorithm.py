import cv2
import matplotlib.pyplot as plt

# Load the grayscale image
img = cv2.imread("/Users/akhi/Desktop/DIP/images/people.png", 0)

# Apply Gaussian Blur to reduce noise
blurred_img = cv2.GaussianBlur(img, (5, 5), 1.4)

# Perform Canny Edge Detection
edges = cv2.Canny(blurred_img, threshold1=50, threshold2=150)

# Display the results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')

plt.show()
