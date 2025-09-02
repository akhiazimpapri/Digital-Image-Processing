import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    # Load image
    img = cv2.imread('/Users/akhi/Desktop/DIP/images/img.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Split channels
    r_channel = img[:, :, 0] #red channel
    g_channel = img[:, :, 1] #green channel
    b_channel = img[:, :, 2] #blue channel

    # Show original + histograms
    show_histograms(img, r_channel, g_channel, b_channel)
    plt.show()

def compute_histogram(channel):
    """Manual histogram (loop method)"""
    h, w = channel.shape
    pixel_array = np.zeros(256, dtype=int)

    for i in range(h):
        for j in range(w):
            pixel_value = channel[i, j]
            pixel_array[pixel_value] += 1
    return pixel_array

def show_histograms(img, r, g, b):
    # Compute histograms
    hist_r = compute_histogram(r)
    hist_g = compute_histogram(g)
    hist_b = compute_histogram(b)

    # Create figure
    plt.figure(figsize=(14, 6))

    # Show original image
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    # Histogram - Red channel
    plt.subplot(2, 2, 2)
    plt.bar(np.arange(256), hist_r, color='r', width=1.0)
    plt.title("Histogram - Red Channel")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")

    # Histogram - Green channel
    plt.subplot(2, 2, 3)
    plt.bar(np.arange(256), hist_g, color='g', width=1.0)
    plt.title("Histogram - Green Channel")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")

    # Histogram - Blue channel
    plt.subplot(2, 2, 4)
    plt.bar(np.arange(256), hist_b, color='b', width=1.0)
    plt.title("Histogram - Blue Channel")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
if __name__ == '__main__':
    main()
