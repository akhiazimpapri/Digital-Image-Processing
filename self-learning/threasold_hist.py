import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('/Users/akhi/Desktop/DIP/images/FLOWER.jpeg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error loading image")
    exit()

# Thresholding functions
def unit_step_threshold(img, t):
    out = np.zeros_like(img)
    out[img >= t] = 255
    return out

def ramp_threshold(img, t):
    out = np.zeros_like(img)
    mask = img >= t
    out[mask] = img[mask] - t
    return out

def quadratic_transform(img, t):
    img_f = img.astype(np.float32)
    trans = 7 * (img_f ** 2) + 3 * img_f - 10
    trans = (trans - trans.min()) / (trans.max() - trans.min()) * 255
    trans = trans.astype(np.uint8)
    out = np.zeros_like(trans)
    out[trans >= t] = 255
    return out

# Apply transformations
u1 = unit_step_threshold(image, 50)
u2 = unit_step_threshold(image, 128)
u3 = unit_step_threshold(image, 200)

r1 = ramp_threshold(image, 50)
r2 = ramp_threshold(image, 100)
r3 = ramp_threshold(image, 150)

q1 = quadratic_transform(image, 50)
q2 = quadratic_transform(image, 128)
q3 = quadratic_transform(image, 200)

# List of images and titles
images = [image, u1, u2, u3, r1, r2, r3, q1, q2, q3]
titles = ['Original', 'Unit Step T=50', 'Unit Step T=128', 'Unit Step T=200',
          'Ramp T=50', 'Ramp T=100', 'Ramp T=150',
          'Quadratic T=50', 'Quadratic T=128', 'Quadratic T=200']

# Plot images
plt.figure(figsize=(15, 12))
plt.suptitle('Images', fontsize=18)
for i in range(len(images)):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot histograms separately
plt.figure(figsize=(15, 12))
plt.suptitle('Histograms', fontsize=18)
for i in range(len(images)):
    plt.subplot(2, 5, i+1)
    plt.hist(images[i].ravel(), bins=256, range=(0, 255), color='black')
    plt.title(titles[i])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
