import matplotlib.pyplot as plt
import cv2 
import numpy as np

def main():
    # Power-law transformation: s = c * r^a
    c = 1.0
    gamma_values = [0.1, 0.3, 0.7, 1, 2, 3]
    img_path = '/Users/akhi/Desktop/akhi/DIP/SC.jpg'

    #Read grayscale image
    imge = cv2.imread(img_path)
    img_gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
    
    if img_gray is None:
        print("Error: Image not found or path is incorrect!")
        return

    print("Image shape:", img_gray.shape)

    # Normalize to [0,1] for gamma correction
    r = img_gray / 255.0

    img_set = [img_gray]  # original image first
    title_set = ['Original']

    for a in gamma_values:
        s = c * (r ** a)  # Apply s = c * r^a
        s_img = np.clip(s * 255, 0, 255).astype(np.uint8)
        img_set.append(s_img)
        title_set.append(f's = c·r^{a}')

    display_imgset(img_set, title_set, row=2, col=4)

def display_imgset(img_set, title_set, row=1, col=1):
    plt.figure(figsize=(10, 10))
    for i in range(len(img_set)):
        plt.subplot(row, col, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(title_set[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
