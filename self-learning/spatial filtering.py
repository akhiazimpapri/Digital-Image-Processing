import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("/Users/akhi/Desktop/DIP/images/SC.jpg", cv2.IMREAD_GRAYSCALE)

# ---------------------------
# 1. Smoothing (Average) Kernel
# ---------------------------
kernel_avg = np.ones((3, 3), np.float32) / 9
img_avg = cv2.filter2D(img, -1, kernel_avg)

# ---------------------------
# 2. Sobel Kernels
# ---------------------------
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

img_sobel_x = cv2.filter2D(img, -1, sobel_x)
img_sobel_y = cv2.filter2D(img, -1, sobel_y)

# ---------------------------
# 3. Prewitt Kernels
# ---------------------------
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float32)

img_prewitt_x = cv2.filter2D(img, -1, prewitt_x)
img_prewitt_y = cv2.filter2D(img, -1, prewitt_y)

# ---------------------------
# 4. Laplace Kernel
# ---------------------------
laplace = np.array([[ 0, -1,  0],
                    [-1,  4, -1],
                    [ 0, -1,  0]], dtype=np.float32)

img_laplace = cv2.filter2D(img, -1, laplace)

# ---------------------------
# Display Results
# ---------------------------
titles = ['Original',
          'Average Blur',
          'Sobel X', 'Sobel Y',
          'Prewitt X', 'Prewitt Y',
          'Laplace']

images = [img, img_avg,
          img_sobel_x, img_sobel_y,
          img_prewitt_x, img_prewitt_y,
          img_laplace]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
