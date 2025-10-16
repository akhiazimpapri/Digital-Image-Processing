import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found. Make sure 'image.png' is in your working directory.")
else:
    # Bit-plane extraction
    bit0 = np.where(cv2.bitwise_and(img, 1 << 0) > 0, 255, 0).astype(np.uint8)
    bit1 = np.where(cv2.bitwise_and(img, 1 << 1) > 0, 255, 0).astype(np.uint8)
    bit2 = np.where(cv2.bitwise_and(img, 1 << 2) > 0, 255, 0).astype(np.uint8)
    bit3 = np.where(cv2.bitwise_and(img, 1 << 3) > 0, 255, 0).astype(np.uint8)
    bit4 = np.where(cv2.bitwise_and(img, 1 << 4) > 0, 255, 0).astype(np.uint8)
    bit5 = np.where(cv2.bitwise_and(img, 1 << 5) > 0, 255, 0).astype(np.uint8)
    bit6 = np.where(cv2.bitwise_and(img, 1 << 6) > 0, 255, 0).astype(np.uint8)
    bit7 = np.where(cv2.bitwise_and(img, 1 << 7) > 0, 255, 0).astype(np.uint8)

    # Reconstruct image from all bit-planes
    reconstructed = (
        (bit0 // 255) * 1 +
        (bit1 // 255) * 2 +
        (bit2 // 255) * 4 +
        (bit3 // 255) * 8 +
        (bit4 // 255) * 16 +
        (bit5 // 255) * 32 +
        (bit6 // 255) * 64 +
        (bit7 // 255) * 128
    ).astype(np.uint8)

    # Display all results
    plt.figure(figsize=(15, 8))

    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(bit0, cmap='gray')
    plt.title('Bit Plane 0 (LSB)')
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.imshow(bit1, cmap='gray')
    plt.title('Bit Plane 1')
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.imshow(bit2, cmap='gray')
    plt.title('Bit Plane 2')
    plt.axis('off')

    plt.subplot(3, 4, 5)
    plt.imshow(bit3, cmap='gray')
    plt.title('Bit Plane 3')
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(bit4, cmap='gray')
    plt.title('Bit Plane 4')
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(bit5, cmap='gray')
    plt.title('Bit Plane 5')
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.imshow(bit6, cmap='gray')
    plt.title('Bit Plane 6')
    plt.axis('off')

    plt.subplot(3, 4, 9)
    plt.imshow(bit7, cmap='gray')
    plt.title('Bit Plane 7 (MSB)')
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
