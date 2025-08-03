import numpy as np
import matplotlib.pyplot as plt

def main():
    h, w = 200, 200

    # Color images
    black_img = np.zeros((h, w, 3), dtype=np.uint8)
    gray1 = np.full((h, w, 3), 50, dtype=np.uint8)
    gray2 = np.full((h, w, 3), 127, dtype=np.uint8)
    gray3 = np.full((h, w, 3), 180, dtype=np.uint8)
    gray4 = np.full((h, w, 3), 220, dtype=np.uint8)
    white_img = np.full((h, w, 3), 255, dtype=np.uint8)

    red_img = np.zeros((h, w, 3), dtype=np.uint8)
    red_img[:, :, 0] = 255

    green_img = np.zeros((h, w, 3), dtype=np.uint8)
    green_img[:, :, 1] = 255

    blue_img = np.zeros((h, w, 3), dtype=np.uint8)
    blue_img[:, :, 2] = 255
    
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    img1[:, :, 0] = 255
    img1[:, :, 1] = 255
    img1[:, :, 2] = 0
    
    img2 = np.zeros((h, w, 3), dtype=np.uint8)
    img2[:, :, 0] = 0
    img2[:, :, 1] = 255
    img2[:, :, 2] = 255
    
    img3 = np.zeros((h, w, 3), dtype=np.uint8)
    img3[:, :, 0] = 255
    img3[:, :, 1] = 0
    img3[:, :, 2] = 255

    img_set = [black_img, gray1, gray2, gray3, gray4, white_img, red_img, green_img, blue_img, img1, img2, img3]
    title_set = ['Black', 'Gray 50', 'Gray 127', 'Gray 180', 'Gray 220', 'White', 'Red', 'Green', 'Blue', '2channel255RG', '2channel255GB','2channel255RB']
    color_set = [None] * len(img_set)

    display_imgset(img_set, color_set, title_set, row=3, col=4)

def display_imgset(img_set, color_set, title_set='', row=1, col=1):
    plt.figure(figsize=(12, 8))
    for k in range(len(img_set)):
        plt.subplot(row, col, k + 1)
        img = img_set[k]
        if color_set[k]:
            plt.imshow(img, cmap=color_set[k], vmin=0, vmax=255)
        else:
            plt.imshow(img)
        plt.title(title_set[k])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
