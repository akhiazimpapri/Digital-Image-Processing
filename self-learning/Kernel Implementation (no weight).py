import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    # Read the image in grayscale
    img_gray = cv2.imread("/Users/akhi/Desktop/DIP/images/FLOWER.jpeg", 0)
    if img_gray is None:
        print("Image not found!")
        return

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
    avg_cv = cv2.filter2D(noisy_img, -1, avg_filter)
    sobel_x_cv = cv2.filter2D(noisy_img, -1, sobel_x)
    sobel_y_cv = cv2.filter2D(noisy_img, -1, sobel_y)
    prewitt_x_cv = cv2.filter2D(noisy_img, -1, prewitt_x)
    prewitt_y_cv = cv2.filter2D(noisy_img, -1, prewitt_y)
    laplace_4_cv = cv2.filter2D(noisy_img, -1, laplace_4)
    laplace_8_cv = cv2.filter2D(noisy_img, -1, laplace_8)

    # Apply manual filters (using cv2.filter2D for simplicity)
    avg_manual = manual_filter(noisy_img, avg_filter)
    sobel_x_manual = manual_filter(noisy_img, sobel_x)
    sobel_y_manual = manual_filter(noisy_img, sobel_y)
    prewitt_x_manual = manual_filter(noisy_img, prewitt_x)
    prewitt_y_manual = manual_filter(noisy_img, prewitt_y)
    laplace_4_manual = manual_filter(noisy_img, laplace_4)
    laplace_8_manual = manual_filter(noisy_img, laplace_8)

    # Collect images for display
    img_set = [
        img_gray, noisy_img,
        avg_cv, avg_manual,
        sobel_x_cv, sobel_x_manual,
        sobel_y_cv, sobel_y_manual,
        prewitt_x_cv, prewitt_x_manual,
        prewitt_y_cv, prewitt_y_manual,
        laplace_4_cv, laplace_4_manual,
        laplace_8_cv, laplace_8_manual
    ]

    img_titles = [
        'Original', 'Gaussian Noise',
        'Avg (cv2)', 'Avg (manual)',
        'Sobel-X (cv2)', 'Sobel-X (manual)',
        'Sobel-Y (cv2)', 'Sobel-Y (manual)',
        'Prewitt-X (cv2)', 'Prewitt-X (manual)',
        'Prewitt-Y (cv2)', 'Prewitt-Y (manual)',
        'Laplace-4 (cv2)', 'Laplace-4 (manual)',
        'Laplace-8 (cv2)', 'Laplace-8 (manual)'
    ]

    display_images(img_set, img_titles)


def manual_filter(img, kernel):
    """Manual convolution using cv2.filter2D."""
    img_float = img.astype(np.float32)
    # cv2.filter2D does convolution; kernel flip not required
    output = cv2.filter2D(img_float, -1, kernel)
    return np.clip(output, 0, 255).astype(np.uint8)


def display_images(img_set, titles):
    plt.figure(figsize=(8, 8))
    n = len(img_set)
    cols = 4
    rows = n // cols + (n % cols != 0)
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(titles[i], fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
