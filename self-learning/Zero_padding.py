import matplotlib.pyplot as plt
import cv2
import numpy as np

def resize_gray_img(img, new_height, new_width):
    old_height, old_width = img.shape
    resized_img = np.zeros((new_height, new_width), dtype=img.dtype)

# These compute the scaling ratios for height and width.
# This tells the function how much to scale the pixel positions.
# 📌 Example:
# If old height is 10 and new height is 5 → each new pixel should represent 2 rows in the original image.

    row_ratio = old_height / new_height
    col_ratio = old_width / new_width

    for i in range(new_height):
        for j in range(new_width):
            old_i = int(i * row_ratio)
            old_j = int(j * col_ratio)
            resized_img[i, j] = img[old_i, old_j]

    return resized_img

def zero_padding(image, pad):
    h, w = image.shape
    padded_img = np.zeros((h + 2*pad, w + 2*pad), dtype=image.dtype)

    for i in range(h):
        for j in range(w):
            padded_img[i + pad][j + pad] = image[i][j]

    return padded_img

def main():
    img_path = '/Users/akhi/Desktop/akhi/DIP/img.png'
    img_path1 = '/Users/akhi/Desktop/akhi/DIP/FLOWER.jpeg'

    # Load and convert to grayscale
    img = cv2.imread(img_path)
    img1 = cv2.imread(img_path1)
    print(img.shape)

    img_gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    print("Image 1 shape before resize:", img_gray0.shape)
    print("Image 2 shape before resize:", img_gray1.shape)

    # Resize manually
    resized_gray0 = resize_gray_img(img_gray0, 1300, 1700)
    resized_gray1 = resize_gray_img(img_gray1, 1300, 1700)

    # Zero padding
    padded_img0 = zero_padding(resized_gray0, pad=100)
    padded_img1 = zero_padding(resized_gray1, pad=100)
    
    print("Image 1 shape after resize:", resized_gray0.shape)
    print("Image 2 shape after resize:", resized_gray1.shape)

    # Plotting
    plt.figure(figsize=(8, 8))

    plt.subplot(3, 2, 1)
    plt.title("Original Image 1")
    plt.imshow(img_gray0, cmap='gray')

    plt.subplot(3, 2, 2)
    plt.title("Original Image 2")
    plt.imshow(img_gray1, cmap='gray')
    
    plt.subplot(3, 2, 3)
    plt.title("Resized Image 1")
    plt.imshow(resized_gray0, cmap='gray')
    
    plt.subplot(3, 2, 4)
    plt.title("Resized Image 2")
    plt.imshow(resized_gray1, cmap='gray')

    plt.subplot(3, 2, 5)
    plt.title("Padded Image 1")
    plt.imshow(padded_img0, cmap='gray')

    plt.subplot(3, 2, 6)
    plt.title("Padded Image 2")
    plt.imshow(padded_img1, cmap='gray')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
