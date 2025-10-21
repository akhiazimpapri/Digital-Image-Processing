import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    # Read the image in grayscale
    img_gray = cv2.imread("/Users/akhi/Desktop/DIP/images/FLOWER.jpeg", 0)
    if img_gray is None:
        raise FileNotFoundError("Image not found! Check the path carefully.")

    # Add Gaussian noise
    row, col = img_gray.shape
    mean, var = 0, 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_img = img_gray.astype(np.float32) + gauss * 255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # Define kernels
    avg_filter = np.ones((3, 3), dtype=np.float32) / 9

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float32)

    laplace_4 = np.array([[ 0, -1,  0],
                          [-1,  4, -1],
                          [ 0, -1,  0]], dtype=np.float32)
    laplace_8 = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)

    # Apply OpenCV filters
    avg = cv2.filter2D(noisy_img, -1, avg_filter)
    sobel_x_f = cv2.filter2D(noisy_img, -1, sobel_x)
    sobel_y_f = cv2.filter2D(noisy_img, -1, sobel_y)
    prewitt_x_f = cv2.filter2D(noisy_img, -1, prewitt_x)
    prewitt_y_f = cv2.filter2D(noisy_img, -1, prewitt_y)
    laplace_4_f = cv2.filter2D(noisy_img, -1, laplace_4)
    laplace_8_f = cv2.filter2D(noisy_img, -1, laplace_8)

    # Apply manual filters
    avg_m = manual_filter(noisy_img, avg_filter)
    sobel_x_m = manual_filter(noisy_img, sobel_x)
    sobel_y_m = manual_filter(noisy_img, sobel_y)
    prewitt_x_m = manual_filter(noisy_img, prewitt_x)
    prewitt_y_m = manual_filter(noisy_img, prewitt_y)
    laplace_4_m = manual_filter(noisy_img, laplace_4)
    laplace_8_m = manual_filter(noisy_img, laplace_8)

    # Collect images for display
    img_set = [
        img_gray, noisy_img,
        avg, avg_m,
        sobel_x_f, sobel_x_m,
        sobel_y_f, sobel_y_m,
        prewitt_x_f, prewitt_x_m,
        prewitt_y_f, prewitt_y_m,
        laplace_4_f, laplace_4_m,
        laplace_8_f, laplace_8_m
    ]
    img_title = [
        'Original', 'Gaussian Noise',
        'Avg (cv2)', 'Avg (manual)',
        'Sobel-X (cv2)', 'Sobel-X (manual)',
        'Sobel-Y (cv2)', 'Sobel-Y (manual)',
        'Prewitt-X (cv2)', 'Prewitt-X (manual)',
        'Prewitt-Y (cv2)', 'Prewitt-Y (manual)',
        'Laplace-4 (cv2)', 'Laplace-4 (manual)',
        'Laplace-8 (cv2)', 'Laplace-8 (manual)'
    ]

    display(img_set, img_title)


def manual_filter(input_img, kernel):
    """Manual convolution using OpenCV backend (efficient method)."""
    tmp_img = input_img.astype(np.float32)
    kernel_flipped = np.flipud(np.fliplr(kernel))
    output_img = cv2.filter2D(tmp_img, -1, kernel_flipped)
    return np.clip(output_img, 0, 255).astype(np.uint8)


def display(img_set, img_title):
    plt.figure(figsize=(8, 8))
    for i in range(len(img_set)):
        plt.subplot(5, 4, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(img_title[i], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
