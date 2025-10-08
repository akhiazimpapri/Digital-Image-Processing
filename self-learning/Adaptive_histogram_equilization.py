import cv2
import numpy as np
import matplotlib.pyplot as plt

def adaptive_histogram_equalization(img, n_parts=8):
    """
    Perform Adaptive Histogram Equalization (AHE) with bilinear interpolation,
    dividing the image into 'n_parts' total regions (approx).
    
    Parameters:
        img (numpy.ndarray): Input grayscale image (0–255)
        n_parts (int): Approximate total number of regions (tiles)
        
    Returns:
        numpy.ndarray: AHE enhanced image
    """
    img = img.astype(np.float32)
    h, w = img.shape
    
    # Determine tile grid based on desired number of parts
    # Example: 8 parts → 2 x 4 grid
    n_tiles_y = int(np.sqrt(n_parts / (w / h)))
    n_tiles_x = int(n_parts / n_tiles_y)
    
    tile_h = h // n_tiles_y
    tile_w = w // n_tiles_x
    
    eq_tiles = np.zeros((n_tiles_y, n_tiles_x, 256), dtype=np.float32)
    
    # Step 1: Compute equalization per tile
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y1 = ty * tile_h
            y2 = (ty + 1) * tile_h if ty < n_tiles_y - 1 else h
            x1 = tx * tile_w
            x2 = (tx + 1) * tile_w if tx < n_tiles_x - 1 else w
            
            tile = img[y1:y2, x1:x2].astype(np.uint8)
            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
            cdf = hist.cumsum()
            cdf = cdf / cdf[-1]  # normalize
            eq_tiles[ty, tx, :] = cdf

    # Step 2: Bilinear interpolation between tiles
    result = np.zeros_like(img)
    
    for y in range(h):
        for x in range(w):
            ty = y / tile_h - 0.5
            tx = x / tile_w - 0.5
            
            ty1 = int(np.floor(ty))
            tx1 = int(np.floor(tx))
            ty2 = ty1 + 1
            tx2 = tx1 + 1
            
            # Clamp
            ty1 = np.clip(ty1, 0, n_tiles_y - 1)
            tx1 = np.clip(tx1, 0, n_tiles_x - 1)
            ty2 = np.clip(ty2, 0, n_tiles_y - 1)
            tx2 = np.clip(tx2, 0, n_tiles_x - 1)
            
            dy = ty - ty1
            dx = tx - tx1
            
            pixel_val = int(img[y, x])
            
            # Interpolate
            cdf_tl = eq_tiles[ty1, tx1, pixel_val]
            cdf_tr = eq_tiles[ty1, tx2, pixel_val]
            cdf_bl = eq_tiles[ty2, tx1, pixel_val]
            cdf_br = eq_tiles[ty2, tx2, pixel_val]
            
            top = (1 - dx) * cdf_tl + dx * cdf_tr
            bottom = (1 - dx) * cdf_bl + dx * cdf_br
            cdf_val = (1 - dy) * top + dy * bottom
            
            result[y, x] = cdf_val * 255.0

    return result.astype(np.uint8)


# ------------------ MAIN SCRIPT ------------------

# Load image in grayscale
img = cv2.imread('/Users/akhi/Desktop/DIP/images/flower.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found! Please check the path.")

# Apply custom AHE dividing into 8 parts
ahe_img = adaptive_histogram_equalization(img, n_parts=64)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('AHE (Divided into 8 Parts with Bilinear Interpolation)')
plt.imshow(ahe_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
