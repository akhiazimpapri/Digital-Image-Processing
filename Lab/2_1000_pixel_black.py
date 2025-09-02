import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/akhi/Desktop/DIP/images/SC.jpg')

if img is None:
    print("Image not found")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Make a copy of the image to modify
modified_img = img_rgb.copy()

# Get image size to avoid index errors
height, width, _ = modified_img.shape

# Loop over top-left 1000x1000 pixels and set them to black
for i in range(min(1000, height)):
    for j in range(min(1000, width)):
        modified_img[i][j] = [0, 0, 0]  # RGB = Black

# Show the original and modified images
plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Top-left 1000x1000 Black")
plt.imshow(modified_img)
plt.axis('off')

plt.tight_layout()
plt.show()
