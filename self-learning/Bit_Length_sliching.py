import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded
if img is None:
    print("Image not found. Make sure 'image.png' is in your working directory.")
else:
    # Create bit-plane images
    bit_planes = []
    for i in range(8):
        plane = cv2.bitwise_and(img, 1 << i)
        plane = np.where(plane > 0, 255, 0).astype(np.uint8)
        bit_planes.append(plane)

    # Plot original image and bit-planes (LSB to MSB)
    plt.figure(figsize=(15, 6))

    # Original image
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Bit planes from 0 (LSB) to 7 (MSB)
    for i in range(8):
        plt.subplot(3, 3, i + 2)
        plt.imshow(bit_planes[i], cmap='gray')
        plt.title(f'Bit Plane {i} (LSB)' if i == 0 else f'Bit Plane {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
