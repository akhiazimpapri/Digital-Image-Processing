import matplotlib.pyplot as plt
import cv2 
import numpy as np

def main():
    img_path = '/Users/akhi/Desktop/akhi/DIP/FLOWER.jpeg'  # Provide the actual image path

    img_3D = cv2.imread(img_path)
    if img_3D is None:
        print("Error: Image not found at the given path.")
        return
    
    img_3D = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    print("Image Shape:", img_3D.shape)

    # # Convert to grayscale for histogram
    # gray_img = cv2.cvtColor(img_3D, cv2.COLOR_RGB2GRAY)

    # Visualization
    plt.figure(figsize=(10, 10))

    # Original Image
    plt.subplot(3, 2, 1)
    plt.title("Main Picture")
    plt.imshow(img_3D)
    plt.axis('off')

    # Red Channel
    plt.subplot(3, 2, 2)
    plt.title("Red Channel")
    plt.imshow(img_3D[:, :, 0], cmap='Reds')
    plt.axis('off')

    # Green Channel
    plt.subplot(3, 2, 3)
    plt.title("Green Channel")
    plt.imshow(img_3D[:, :, 1], cmap='Greens')
    plt.axis('off')

    # Blue Channel
    plt.subplot(3, 2, 4)
    plt.title("Blue Channel")
    plt.imshow(img_3D[:, :, 2], cmap='Blues')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    def prepare_histogram(img_3D):
        h, w, channels = img_3D.shape
    pixel_count = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            pixel_value = img_3D[i, j]
            pixel_count[pixel_value] += 1

    print(pixel_count)  # Optional: check values

    # Plot histogram
    x = np.arange(256)
    plt.figure(figsize=(6, 4))
    plt.plot(x, pixel_count, 'ro')
    plt.title("Grayscale Histogram (Manual Count)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()
    
    # Call histogram function
    prepare_histogram(img_3D)

if __name__ == '__main__':
    main()
