import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- MODIFY THESE PATHS if needed ---
path_large = '/Users/akhi/Desktop/DIP/images/img.png'      # larger image
path_small = '/Users/akhi/Desktop/DIP/images/FLOWER.jpeg' # smaller image
# ------------------------------------

# Check files exist
for p,name in [(path_large,'large'), (path_small,'small')]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{name} image not found: {p!r}")

# Read images (color) and check
large = cv2.imread(path_large, cv2.IMREAD_COLOR)
small = cv2.imread(path_small, cv2.IMREAD_COLOR)
if large is None or small is None:
    raise ValueError("cv2.imread returned None. Check the file path and image permissions.")

# Convert to grayscale (safe conversion in case already single-channel)
if large.ndim == 3:
    large_gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
else:
    large_gray = large.copy()
if small.ndim == 3:
    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
else:
    small_gray = small.copy()

h2, w2 = large_gray.shape
h1, w1 = small_gray.shape
print(f"Large: {w2}x{h2}, dtype={large_gray.dtype}; Small: {w1}x{h1}, dtype={small_gray.dtype}")

# If small is bigger than large, resize small to fit while keeping aspect ratio
if h1 > h2 or w1 > w2:
    scale = min(h2 / h1, w2 / w1)
    new_w = max(1, int(round(w1 * scale)))
    new_h = max(1, int(round(h1 * scale)))
    print(f"Small image is larger than large canvas -> resizing to {new_w}x{new_h}")
    small_gray = cv2.resize(small_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    h1, w1 = small_gray.shape

# Create black canvas (zeros) with same dtype and shape as large
result_black = np.zeros((h2, w2), dtype=np.uint8)

# Compute centered top-left coordinate (safe, non-negative)
start_y = (h2 - h1) // 2
start_x = (w2 - w1) // 2
print(f"Placing small at y={start_y}, x={start_x}")

# Place small image into the black canvas (centered)
result_black[start_y:start_y + h1, start_x:start_x + w1] = small_gray

# Alternative: paste small ON TOP of the large image (keeps large background)
result_on_large = large_gray.copy()
result_on_large[start_y:start_y + h1, start_x:start_x + w1] = small_gray

# Show results
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.title("Large (gray)")
plt.imshow(large_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Small (gray)")
plt.imshow(small_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Black canvas with Small centered")
plt.imshow(result_black, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Small pasted onto Large")
plt.imshow(result_on_large, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
