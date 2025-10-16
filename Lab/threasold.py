import numpy as np
import matplotlib.pyplot as plt
import cv2

def func1(img):
    m = 128
    img1 = np.where(img >= m, 1, 0).astype(np.uint8) * 255
    return img1

def func2(img):
    m = 1.02
    c = 0
    thres1 = 128
    thres2 = 196
    img1 = np.zeros_like(img, dtype=np.float32)
    
    img1[img < thres1] = m * img[img < thres1] + c
    mask = (img >= thres1) & (img < thres2)
    img1[mask] = 0.5 * 255
    img1[img >= thres2] = 255
    
    return np.clip(img1, 0, 255).astype(np.uint8)

def func3(img):
    m = 1.05
    c = 5
    thres1 = 50
    thres2 = 196
    img1 = np.zeros_like(img, dtype=np.float32)
    
    img1[img <= thres1] = 0
    mask = (img > thres1) & (img <= thres2)
    img1[mask] = m * img[mask] + c
    img1[img > thres2] = 0.75 * 255
    
    return np.clip(img1, 0, 255).astype(np.uint8)

def main():
    img1 = cv2.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg', 0)
    if img1 is None:
        print("Image not found!")
        return
    
    img11 = func1(img1)
    img12 = func2(img1)
    img13 = func3(img1)
    
    img_set = [img1, img11, img12, img13]
    title_set = ["Original image", "Applying step func", "Applying func2", "Applying func3"]
    
    plt.figure(figsize=(15, 8))
    
    for i in range(4):
        plt.subplot(2, 4, i+1)
        plt.title(title_set[i])
        plt.imshow(img_set[i], cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 4, i+5)
        plt.hist(img_set[i].ravel(), bins=256, color='gray')
        plt.title("Histogram")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
