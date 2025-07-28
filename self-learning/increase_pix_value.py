import matplotlib.pyplot as plt
import cv2 

def main():
    C = 10
    img_path = '/Users/akhi/Desktop/akhi/DIP/FLOWER.jpeg'  # Provide the actual image path here

    img_3D = cv2.imread(img_path)
    if img_3D is None:
        print("Error: Image not found at the given path.")
        return
    
    img_3D = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    print("Image Shape:", img_3D.shape)

    extra_pixel = img_3D + C

    plt.figure(figsize=(8, 8))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.title("Main Image")
    plt.imshow(img_3D)

    plt.subplot(2, 2, 2)
    plt.title("Output Image")
    plt.imshow(extra_pixel)
    
    # --- Histogram of Original Image ---
    plt.subplot(2, 2, 3)
    plt.title("Histogram: Original")
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        plt.hist(img_3D[:, :, i].ravel(), bins=256, range=(0, 256), color=color, alpha=0.5, label=f'{color.upper()}')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    # --- Histogram of Brightened Image ---
    plt.subplot(2, 2, 4)
    plt.title("Histogram: Brightened Image")
    for i, color in enumerate(colors):
        plt.hist(extra_pixel[:, :, i].ravel(), bins=256, range=(0, 256), color=color, alpha=0.5, label=f'{color.upper()}')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    
    #plt.tight_layout()  # Auto-adjusts spacing

    plt.show()
    plt.close()
    plt.axis()

if __name__ == '__main__':
    main()
 