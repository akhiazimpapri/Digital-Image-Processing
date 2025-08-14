import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error loading image")
    exit()

# Unit Step Thresholding
def unit_step_threshold(img, t):
    out = np.zeros_like(img)
    out[img >= t] = 255
    return out

# Ramp Thresholding
def ramp_threshold(img, t):
    out = np.zeros_like(img)
    mask = img >= t
    out[mask] = img[mask] - t
    return out

# Quadratic Transform: y = 7x² + 3x - 10
def quadratic_transform(img, t):
    img_f = img.astype(np.float32)
    trans = 7 * (img_f ** 2) + 3 * img_f - 10
    trans = (trans - trans.min()) / (trans.max() - trans.min()) * 255
    trans = trans.astype(np.uint8)
    out = np.zeros_like(trans)
    out[trans >= t] = 255
    return out

# Apply transformations (no loops)
u1 = unit_step_threshold(image, 50)
u2 = unit_step_threshold(image, 128)
u3 = unit_step_threshold(image, 200)

r1 = ramp_threshold(image, 50)
r2 = ramp_threshold(image, 100)
r3 = ramp_threshold(image, 150)

q1 = quadratic_transform(image, 50)
q2 = quadratic_transform(image, 128)
q3 = quadratic_transform(image, 200)

# Plot results with subplot
plt.figure(figsize=(10, 10))
plt.suptitle('Orginal and Thresholded Images', fontsize=18)

# Main original image
plt.subplot(4, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Row 2: Unit Step
plt.subplot(4, 3, 4)
plt.imshow(u1, cmap='gray')
plt.title('Unit Step T=50')
plt.axis('off')

plt.subplot(4, 3, 5)
plt.imshow(u2, cmap='gray')
plt.title('Unit Step T=128')
plt.axis('off')

plt.subplot(4, 3, 6)
plt.imshow(u3, cmap='gray')
plt.title('Unit Step T=200')
plt.axis('off')

# Row 3: Ramp
plt.subplot(4, 3, 7)
plt.imshow(r1, cmap='gray')
plt.title('Ramp T=50')
plt.axis('off')

plt.subplot(4, 3, 8)
plt.imshow(r2, cmap='gray')
plt.title('Ramp T=100')
plt.axis('off')

plt.subplot(4, 3, 9)
plt.imshow(r3, cmap='gray')
plt.title('Ramp T=150')
plt.axis('off')

# Row 4: Quadratic
plt.subplot(4, 3, 10)
plt.imshow(q1, cmap='gray')
plt.title('Quadratic T=50')
plt.axis('off')

plt.subplot(4, 3, 11)
plt.imshow(q2, cmap='gray')
plt.title('Quadratic T=128')
plt.axis('off')

plt.subplot(4, 3, 12)
plt.imshow(q3, cmap='gray')
plt.title('Quadratic T=200')
plt.axis('off')

plt.tight_layout()
plt.show()
