import matplotlib.pyplot as plt
import numpy as np
import cv2

# --- Function to display image set ---
def display_imgset(img_set, color_set, title_set = '', row = 1, col = 1):
    plt.figure(figsize = (20, 20))
    k = 1
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            plt.subplot(row, col, k)
            img = img_set[k - 1]
            if len(img.shape) == 3:
                plt.imshow(img)
            else:
                plt.imshow(img, cmap = color_set[k - 1])
            if title_set[k - 1] != '':
                plt.title(title_set[k - 1])
            k += 1
    plt.show()
    plt.close()

# --- Function to prepare histogram ---
def prepare_histogram(img, color_channel):
    pixel_count = np.zeros((256,), dtype=np.uint64)
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            pixel_count[pixel_value] += 1

    print(pixel_count)

    x = np.arange(256)
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x, pixel_count, 'ro')
    plt.title('Histogram of ' + color_channel + ' Channel')
    plt.xlabel('Pixel Values')
    plt.ylabel('Number of Pixels')

    plt.subplot(1, 2, 2)
    plt.bar(x, pixel_count)
    plt.title('Histogram of ' + color_channel + ' Channel')
    plt.xlabel('Pixel Values')
    plt.ylabel('Number of Pixels')

    plt.show()
    plt.close()

# --- Main Execution ---

# Load image in BGR, convert to RGB
img = cv2.imread('/Users/akhi/Desktop/akhi/DIP/FLOWER.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Split RGB channels
red_img = img_rgb[:, :, 0]
green_img = img_rgb[:, :, 1]
blue_img = img_rgb[:, :, 2]

# Print pixel matrices
print("RGB Image:\n", img_rgb)
print("Red Channel:\n", red_img)
print("Green Channel:\n", green_img)
print("Blue Channel:\n", blue_img)
print("Top-left 5x5 pixels:\n", img_rgb[:5, :5])

# Convert to grayscale
gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
print("Grayscale Image:\n", gray_img)

# Display image set
img_set = [img_rgb, red_img, green_img, blue_img]
title_set = ['RGB', 'Red', 'Green', 'Blue']
color_set = ['', 'Reds', 'Greens', 'Blues']
display_imgset(img_set, color_set, title_set, row=2, col=2)

# Plot histograms
prepare_histogram(red_img, 'Red')
prepare_histogram(green_img, 'Green')
prepare_histogram(blue_img, 'Blue')
