import matplotlib.pyplot as plt
import cv2 

def main():
    img_path = '/Users/akhi/Desktop/DIP/images/FLOWER.jpeg'
    img_path1 = '/Users/akhi/Desktop/DIP/images/roses.png'

    img_3D = cv2.imread(img_path)
    img_3D1 = cv2.imread(img_path1)
    
    img_3D = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    img_3D1 = cv2.cvtColor(img_3D1, cv2.COLOR_BGR2RGB)
    
    resized = cv2.resize(img_3D, (300,300))
    resized1 = cv2.resize(img_3D1,(300,300))
    
    if img_3D is None:
        print("Error: Image not found at the given path.")
        return
    
    
    print("Image Shape:", img_3D.shape)
    print("Image Shape:", img_3D1.shape)

    extra_pixel = resized + resized1
    
    print("Image Shape:", extra_pixel.shape)

    plt.figure(figsize=(8, 8))

    # Plotting
    plt.subplot(3, 2, 1)
    plt.title("Main Image 1")
    plt.imshow(img_3D)
    
    plt.subplot(3, 2, 2)
    plt.title("Main Image 2")
    plt.imshow(img_3D1)
    
    plt.subplot(3, 2, 3)
    plt.title("resized img 1")
    plt.imshow(resized)
    
    plt.subplot(3, 2, 4)
    plt.title("resized img 2")
    plt.imshow(resized1)

    plt.subplot(3, 2, 5)
    plt.title("Output Image")
    plt.imshow(extra_pixel)
    
    
    plt.tight_layout()  # Auto-adjusts spacing

    plt.show()
    plt.close()
    plt.axis()

if __name__ == '__main__':
    main()
 