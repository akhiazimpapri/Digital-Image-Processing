import cv2
import numpy as np
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Images
# -------------------------------
source = cv2.imread("/Users/akhi/Desktop/DIP/images/flower.png", cv2.IMREAD_GRAYSCALE)      # Source image
reference = cv2.imread("/Users/akhi/Desktop/DIP/images/roses.png", cv2.IMREAD_GRAYSCALE) # Reference image

if source is None or reference is None:
    raise FileNotFoundError("Source or Reference image not found!")

# -------------------------------
# 2. Built-in Histogram Matching (Scikit-Image)
# -------------------------------
matched_builtin = match_histograms(source, reference)

# -------------------------------
# 3. Custom Method 1: CDF-based Histogram Matching
# -------------------------------
def histogram_matching_cdf(source, reference):
    src_hist, _ = np.histogram(source.flatten(), 256, [0,256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0,256])

    src_cdf = np.cumsum(src_hist) / np.sum(src_hist)
    ref_cdf = np.cumsum(ref_hist) / np.sum(ref_hist)

    mapping = np.interp(src_cdf, ref_cdf, np.arange(256))
    matched = mapping[source]
    return matched.astype(np.uint8)

matched_cdf = histogram_matching_cdf(source, reference)

# -------------------------------
# 4. Custom Method 2: Direct Histogram Specification
# -------------------------------
def histogram_matching_direct(source, reference):
    src_values, bin_idx, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    ref_values, ref_counts = np.unique(reference.ravel(), return_counts=True)

    # Cumulative probability
    src_quantiles = np.cumsum(src_counts).astype(np.float64)
    src_quantiles /= src_quantiles[-1]

    ref_quantiles = np.cumsum(ref_counts).astype(np.float64)
    ref_quantiles /= ref_quantiles[-1]

    # Interpolation
    interp_values = np.interp(src_quantiles, ref_quantiles, ref_values)
    return interp_values[bin_idx].reshape(source.shape).astype(np.uint8)

matched_direct = histogram_matching_direct(source, reference)

# -------------------------------
# 5. Contrast Adjustment Function
# -------------------------------
def adjust_contrast(image, alpha, beta):
    """
    alpha < 1.0 = low contrast
    alpha > 1.0 = high contrast
    beta = brightness shift
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

low_contrast = adjust_contrast(source, alpha=0.5, beta=0)
high_contrast = adjust_contrast(source, alpha=2.0, beta=0)
normal_contrast = source.copy()

# -------------------------------
# 6. Show Results
# -------------------------------
plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1); plt.imshow(source, cmap='gray'); plt.title("Source"); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(reference, cmap='gray'); plt.title("Reference"); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(matched_builtin, cmap='gray'); plt.title("Built-in Matched"); plt.axis('off')
plt.subplot(2, 3, 4); plt.imshow(matched_cdf, cmap='gray'); plt.title("Custom CDF Matched"); plt.axis('off')
plt.subplot(2, 3, 5); plt.imshow(matched_direct, cmap='gray'); plt.title("Custom Direct Matched"); plt.axis('off')
plt.subplot(2, 3, 6); plt.hist(matched_builtin.ravel(), bins=256, color='black'); plt.title("Histogram of Built-in")
plt.show()

# -------------------------------
# 7. Effect of Contrast Levels
# -------------------------------
contrast_levels = [("Low Contrast", low_contrast),
                   ("Normal Contrast", normal_contrast),
                   ("High Contrast", high_contrast)]

plt.figure(figsize=(15, 8))
for i, (title, img) in enumerate(contrast_levels):
    matched_img = match_histograms(img, reference)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{title} Source")
    plt.axis('off')
    
    plt.subplot(2, 3, i + 4)
    plt.imshow(matched_img, cmap='gray')
    plt.title(f"{title} Matched")
    plt.axis('off')

plt.show()