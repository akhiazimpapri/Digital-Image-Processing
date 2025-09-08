import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path ='/Users/akhi/Desktop/DIP/images/FLOWER.jpeg '
    
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # img_rgb1 = img_bgr[:, :, ::-1] ## Reverse the last axis (channel axis)
    
    # Manually split channels
    blue_channel = img_bgr[:, :, 0]
    green_channel = img_bgr[:, :, 1]
    red_channel = img_bgr[:, :, 2]

    # Manually combine in RGB order
    img_rgb1 = np.stack((red_channel, green_channel, blue_channel),axis = 2)

    
    plt.figure(figsize=(10, 10))
    # Original Image
    
    plt.subplot(3, 2, 1)
    plt.title("Main Picture loded by cv2(bgr)")
    plt.imshow(img_bgr)
    
    plt.subplot(3, 2, 2)
    plt.title("BGR2RGB by cv2")
    plt.imshow(img_rgb)
    
    plt.subplot(3, 2, 3)
    plt.title("BGR2RGB manualy")
    plt.imshow(img_rgb1)
    
    
    
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