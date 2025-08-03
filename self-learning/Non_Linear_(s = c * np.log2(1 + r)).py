import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    c = 500.0  # constant multiplier
    img_path = '/Users/akhi/Desktop/akhi/DIP/scn.jpeg'
    
    # Read image in grayscale
    img_gray = cv2.imread(img_path,0)
    
    if img_gray is None:
        print("Error: Image not found or path is incorrect!")
        return

    print("Image shape:", img_gray.shape)

    # Normalize to [0, 1]
    r = img_gray / 255.0

    # Apply log2 transformation: s = c * log2(1 + r)
    s = c * np.log2(1 + r)

    # Normalize back to [0, 255]
    s_img = np.clip(s / np.max(s) * 255, 0, 255).astype(np.uint8)

    # Show original and transformed image
    display_imgset([img_gray, s_img], ['Original', 's = c·log₂(1 + r)'], row=1, col=2)

def display_imgset(img_set, title_set, row=1, col=1):
    plt.figure(figsize=(10, 5))
    for i in range(len(img_set)):
        plt.subplot(row, col, i + 1)
        plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])
        plt.axis()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
