import cv2
import numpy as np
import matplotlib.pyplot as plt

# MAIN FUNCTION
def main():
    # Read Image
    img_path = "/Users/akhi/Desktop/DIP/images/paddy.png"
    img_3D = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_3D, cv2.COLOR_BGR2GRAY)

    # Parameters
    print("Image shape:", img_gray.shape)
    cutoff = 40      # Cutoff frequency (radius)
    order = 2        # Butterworth filter order (controls slope)

    # Compute DFT
    dft = np.fft.fft2(img_gray)
    dft_shift = np.fft.fftshift(dft)

    # Butterworth Low-pass Filter
    butter_lp = butterworth_lowpass_mask(img_gray.shape, cutoff, order)
    low_passed = dft_shift * butter_lp
    img_low = np.fft.ifft2(np.fft.ifftshift(low_passed))
    img_low = np.abs(img_low)

    # Butterworth High-pass Filter
    butter_hp = butterworth_highpass_mask(img_gray.shape, cutoff, order)
    high_passed = dft_shift * butter_hp
    img_high = np.fft.ifft2(np.fft.ifftshift(high_passed))
    img_high = np.abs(img_high)

    # Visualization
    plt.figure(figsize=(8, 8))

    plt.subplot(3, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.imshow(np.log(1 + np.abs(dft_shift)), cmap='gray')
    plt.title("Magnitude Spectrum")
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.imshow(butter_lp, cmap='gray')
    plt.title("Butterworth Low-pass Mask")
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.imshow(img_low, cmap='gray')
    plt.title("Low-pass Filtered Image")
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.imshow(butter_hp, cmap='gray')
    plt.title("Butterworth High-pass Mask")
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.imshow(img_high, cmap='gray')
    plt.title("High-pass Filtered Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=True)



# BUTTERWORTH FILTER MASKS

def butterworth_lowpass_mask(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # Create meshgrid of distances from center
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v, u)
    D = np.sqrt((U - ccol) ** 2 + (V - crow) ** 2)

    # Butterworth low-pass transfer function
    H = 1 / (1 + (D / cutoff) ** (2 * order))
    return H


def butterworth_highpass_mask(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v, u)
    D = np.sqrt((U - ccol) ** 2 + (V - crow) ** 2)

    # Butterworth high-pass transfer function
    H = 1 / (1 + (cutoff / (D + 1e-5)) ** (2 * order))  # +1e-5 avoids division by zero
    return H


if __name__ == "__main__":
    main()
