import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('/Users/akhi/Desktop/akhi/DIP/FLOWER.jpeg')
if img is None:
    print("Image not found")
    exit()

# Convert to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Make a copy to modify
modified_img = img_rgb.copy()

# Get height and width
height, width, _ = modified_img.shape

# Calculate center point
center_y = height // 2 #Returns the integer part only
center_x = width // 2

# Define the 100x100 region around the center
start_y = max(center_y - 50, 0)
end_y = min(center_y + 50, height)
start_x = max(center_x - 50, 0)
end_x = min(center_x + 50, width)

# Use nested loop to set pixels to black
for i in range(start_y, end_y):
    for j in range(start_x, end_x):
        modified_img[i][j] = [0, 0, 0]  # Black pixel

# Show the result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Center 100x100 Black (Loop)")
plt.imshow(modified_img)
plt.axis('off')

plt.tight_layout()
plt.show()
