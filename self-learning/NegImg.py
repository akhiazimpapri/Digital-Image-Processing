import matplotlib.pyplot as plt
import cv2 

def main():
    img_path = '/Users/akhi/Desktop/akhi/DIP/img.png'  # Provide the actual image path here

    img_3D = cv2.imread(img_path)
    if img_3D is None:
        print("Error: Image not found at the given path.")
        return
    
    img_3D = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    print("Image Shape:", img_3D.shape)

    # Create negative image
    img_neg = 255 - img_3D

    plt.figure(figsize=(10, 10))

    # Original Image
    plt.subplot(3, 2, 1)
    plt.title("Main Picture")
    plt.imshow(img_3D)

    # Negative Image
    plt.subplot(3, 2, 2)
    plt.title("Negative Picture")
    plt.imshow(img_neg)

    # Red Channel
    plt.subplot(3, 2, 3)
    plt.title("Red Channel")
    plt.imshow(img_3D[:, :, 0], cmap='Reds')

    # Green Channel
    plt.subplot(3, 2, 4)
    plt.title("Green Channel")
    plt.imshow(img_3D[:, :, 1], cmap='Greens')

    # Blue Channel
    plt.subplot(3, 2, 5)
    plt.title("Blue Channel")
    plt.imshow(img_3D[:, :, 2], cmap='Blues')
    
    #plt.tight_layout()  # Auto-adjusts spacing

    plt.show()
    plt.close()
    plt.axis()

if __name__ == '__main__':
    main()
 