import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img_path = '/Users/akhi/Desktop/DIP/images/img.png'
    img_3D = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_3D, cv2.COLOR_BGR2GRAY)

    Filter_ver = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    Filter_hor = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
    Filter_new = np.array([[0, 0, 0],
                           [0, 1/9, 0],
                           [0, 0, 0]])
    Filter_random = np.random.rand(3, 3)
    

    filter_ver = cv2.filter2D(img_gray, -1, Filter_ver)
    filter_hor = cv2.filter2D(img_gray, -1, Filter_hor)
    filter_new = cv2.filter2D(img_gray, -1, Filter_new)
    filter_random = cv2.filter2D(img_gray, -1, Filter_random)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Vertical')
    plt.imshow(filter_ver, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Horizontal')
    plt.imshow(filter_hor, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title('1/9')
    plt.imshow(filter_new, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title('random')
    plt.imshow(filter_random, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
