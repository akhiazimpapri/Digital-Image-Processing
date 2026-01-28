import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# 1. Gaussian Kernel
# --------------------------------
def gaussian_kernel(size, sigma=1.4):
    ax = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

# --------------------------------
# 2. Non-Maximum Suppression
# --------------------------------
def non_max_suppression(mag, angle):
    h, w = mag.shape
    output = np.zeros((h, w), dtype=np.float64)
    angle = np.rad2deg(angle)
    angle[angle < 0] += 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = r = 0

            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if mag[i,j] >= q and mag[i,j] >= r:
                output[i,j] = mag[i,j]

    return output

# --------------------------------
# 3. Double Threshold
# --------------------------------
def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    high = img.max() * high_ratio
    low = high * low_ratio

    strong = 255
    weak = 75

    result = np.zeros(img.shape, dtype=np.uint8)

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, weak, strong

# --------------------------------
# 4. Edge Tracking by Hysteresis
# --------------------------------
def hysteresis(img, weak, strong):
    h, w = img.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == weak:
                if np.any(img[i-1:i+2, j-1:j+2] == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

# --------------------------------
# 5. Main Canny Function
# --------------------------------
def canny_edge_detection(image_path, kernel_size=5):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float
    gray = gray.astype(np.float64)

    # Gaussian smoothing
    kernel = gaussian_kernel(kernel_size)
    smooth = cv2.filter2D(gray, -1, kernel)

    # Sobel filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    sobel_y = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]], dtype=np.float64)

    gx = cv2.filter2D(smooth, -1, sobel_x)
    gy = cv2.filter2D(smooth, -1, sobel_y)

    # Gradient magnitude & direction
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = magnitude / magnitude.max() * 255
    direction = np.arctan2(gy, gx)

    # Non-maximum suppression
    nms = non_max_suppression(magnitude, direction)

    # Double threshold
    thresh, weak, strong = double_threshold(nms)

    # Hysteresis
    edges = hysteresis(thresh, weak, strong)

    return gray.astype(np.uint8), edges

# --------------------------------
# 6. Run & Display
# --------------------------------
if __name__ == "__main__":
    image_path = "/Users/akhi/Desktop/DIP/images/canny.png" 
    gray, edges = canny_edge_detection(image_path, kernel_size=5)

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection Output")
    plt.axis('off')

    plt.show()
