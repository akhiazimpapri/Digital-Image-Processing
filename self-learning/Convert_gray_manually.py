import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = '/Users/akhi/Desktop/akhi/DIP/scn.jpeg'
    
    img_rgb = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    # Read image in grayscale
    img_gray = cv2.imread(img_path,0)
    
    r = 0.299
    g = 0.587
    b = 0.142
    
    x = img_rgb[:, :, 0]
    y = img_rgb[:, :, 1]
    z = img_rgb[:, :, 2]
    
    p = (x*0.299 + y*0.587 + z*0.42)/3
    
    q = (x*(1/3) + y*(1/3) + z*(1/3))
    
    plt.figure(figsize=(10, 10))
    # Original Image
    
    plt.subplot(3, 2, 1)
    plt.title("Main Picture")
    plt.imshow(img_rgb)
    
    plt.subplot(3, 2, 2)
    plt.title("Main Picture gray by cv2")
    plt.imshow(img_gray)
    
    plt.subplot(3, 2, 3)
    plt.title("Main Picture gray manualy non-linear")
    plt.imshow(p)
    
    plt.subplot(3, 2, 4)
    plt.title("Main Picture gray manualy linear")
    plt.imshow(q)
    
    
    
    # plt.subplot(3, 2, 3)
    # plt.title("RED")
    # plt.imshow(x, cmap = 'Reds')
    
    # plt.subplot(3, 2, 4)
    # plt.title("GREEN")
    # plt.imshow(y, cmap = 'Greens')
    
    # plt.subplot(3, 2, 5)
    # plt.title("BLUE")
    # plt.imshow(z, cmap = 'Blues')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()