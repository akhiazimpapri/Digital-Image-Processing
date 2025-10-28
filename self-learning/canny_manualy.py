import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_blur(img, ksize=5, sigma=1.4):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def sobel_gradients(img):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    Gx = cv2.filter2D(img, -1, Kx)
    Gy = cv2.filter2D(img, -1, Ky)

    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / magnitude.max() * 255
    angle = np.arctan2(Gy, Gx)
    return magnitude, angle

def non_max_suppression(magnitude, angle):
    H, W = magnitude.shape
    Z = np.zeros((H,W), dtype=np.int32)
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,H-1):
        for j in range(1,W-1):
            q, r = 255, 255
            # angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                Z[i,j] = magnitude[i,j]
            else:
                Z[i,j] = 0
    return Z

def double_threshold(img, low, high):
    strong = 255
    weak = 50
    res = np.zeros_like(img)

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

def hysteresis(img, weak, strong=255):
    H, W = img.shape
    for i in range(1, H-1):
        for j in range(1, W-1):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    return img

def canny_edge_detector(img, low=50, high=100):
    # Step 1: Gaussian Blur
    blurred = gaussian_blur(img)

    # Step 2: Gradient Magnitude and Direction
    magnitude, angle = sobel_gradients(blurred)

    # Step 3: Non-Max Suppression
    nms = non_max_suppression(magnitude, angle)

    # Step 4: Double Threshold
    dt, weak, strong = double_threshold(nms, low, high)

    # Step 5: Hysteresis
    final = hysteresis(dt, weak, strong)
    return final

# RUN
img = cv2.imread('/Users/akhi/Desktop/DIP/images/people.png', 0)
edges = canny_edge_detector(img, 50, 100)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title("User-defined Canny Edge Detection")
plt.axis('off')

plt.show()
