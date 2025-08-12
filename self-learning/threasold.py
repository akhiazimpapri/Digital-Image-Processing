import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error loading image")
    exit()

# 1. Unit Step Thresholding function
def unit_step_threshold(img, threshold):
    output = np.zeros_like(img)
    output[img >= threshold] = 255
    return output

def unit_step_threshold_three(img):
    thresholds = [50, 128, 200]
    return [unit_step_threshold(img, t) for t in thresholds]

# 2. Ramp Thresholding function
def ramp_threshold(img, threshold):
    output = np.zeros_like(img, dtype=np.uint8)
    mask = img >= threshold
    output[mask] = img[mask] - threshold
    return output

def ramp_threshold_three(img):
    thresholds = [50, 100, 150]
    return [ramp_threshold(img, t) for t in thresholds]

# 3. Power-law (s = c * r^a) transform function
def power_law_transform(img, c, a):
    img_norm = img / 255.0
    s = c * np.power(img_norm, a)
    s = np.clip(s, 0, 1)
    s = (s * 255).astype(np.uint8)
    return s

def power_law_transform_three(img):
    params = [(1, 0.5), (1, 1), (1, 2.0)]  # (c, a)
    return [power_law_transform(img, c, a) for c, a in params]

# Process image with all functions
unit_step_results = unit_step_threshold_three(image)
ramp_results = ramp_threshold_three(image)
power_law_results = power_law_transform_three(image)

# Plotting all results
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Thresholding and Power-law Transformations', fontsize=18)

# Titles for each row
row_titles = ['Unit Step Thresholding', 'Ramp Thresholding', 'Power-law Transform']

# Plot Unit Step Threshold results
for i, img_out in enumerate(unit_step_results):
    axes[0, i].imshow(img_out, cmap='gray')
    axes[0, i].set_title(f'Threshold = {[50,128,200][i]}')
    axes[0, i].axis('off')

# Plot Ramp Threshold results
for i, img_out in enumerate(ramp_results):
    axes[1, i].imshow(img_out, cmap='gray')
    axes[1, i].set_title(f'Threshold = {[50,100,150][i]}')
    axes[1, i].axis('off')

# Plot Power-law Transform results
params = [(1, 0.5), (1, 1), (1, 2.0)]
for i, img_out in enumerate(power_law_results):
    c, a = params[i]
    axes[2, i].imshow(img_out, cmap='gray')
    axes[2, i].set_title(f'c={c}, a={a}')
    axes[2, i].axis('off')

# Add row titles on the left side
for ax, row_title in zip(axes[:,0], row_titles):
    ax.set_ylabel(row_title, rotation=90, size='large', labelpad=15)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
