import matplotlib.pyplot as plt
import cv2
import numpy as np

# -------------------------------------------------
#                MAIN FUNCTION
# -------------------------------------------------
def main():
    """
    Main driver function.
    Steps:
        1. Read and convert the image to grayscale.
        2. Compute DFT and visualize the original magnitude spectrum.
        3. Apply Low-pass, High-pass, and Band-pass filters.
        4. Display results and histograms for each filtered image.
    """
    # ---------- Read Image ----------
    img_path = "/Users/akhi/Desktop/DIP/images/paddy.png"
    img_3D = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_3D, cv2.COLOR_BGR2GRAY)

    # ---------- Parameters ----------
    print("Image shape:", img_gray.shape)
    radius_low = img_gray.shape[0] // 4    # Radius for low-pass filter
    radius_high = img_gray.shape[0] // 6   # Radius for high-pass filter

    # ---------- DFT without filtering ----------
    dft_shift, magnitude_dft_original, img_back_original = dft(img_gray)
    img_set = [img_gray, magnitude_dft_original, img_back_original, histogram(img_gray)]
    img_title = ['Input Image', 'Magnitude Spectrum (Original)', 'Reconstructed Image (Original)', 'Histogram (Original)']
    display(img_set, img_title)

    # ---------- Low-pass Filtering ----------
    img_back_low, magnitude_dft_low_passed = low_pass_filtering(img_gray, dft_shift, radius_low)
    img_set = [img_gray, magnitude_dft_low_passed, img_back_low, histogram(img_back_low)]
    img_title = ['Input Image', 'Magnitude Spectrum (Low-pass)', 'Reconstructed Image (Low-pass)', 'Histogram (Low-pass)']
    display(img_set, img_title)

    # ---------- High-pass Filtering ----------
    img_back_high, magnitude_dft_high_passed = high_pass_filtering(img_gray, dft_shift, radius_high)
    img_set = [img_gray, magnitude_dft_high_passed, img_back_high, histogram(img_back_high)]
    img_title = ['Input Image', 'Magnitude Spectrum (High-pass)', 'Reconstructed Image (High-pass)', 'Histogram (High-pass)']
    display(img_set, img_title)

    # ---------- Band-pass Filtering ----------
    img_back_band, magnitude_dft_band_passed = band_pass_filtering(img_gray, dft_shift, radius_low, radius_high)
    img_set = [img_gray, magnitude_dft_band_passed, img_back_band, histogram(img_back_band)]
    img_title = ['Input Image', 'Magnitude Spectrum (Band-pass)', 'Reconstructed Image (Band-pass)', 'Histogram (Band-pass)']
    display(img_set, img_title)


# -------------------------------------------------
#        DFT and IDFT (without filtering)
# -------------------------------------------------
def dft(img_gray):
    """
    Perform Discrete Fourier Transform (DFT) and its inverse.

    Args:
        img_gray (numpy.ndarray): Grayscale input image.

    Returns:
        tuple:
            dft_shift (numpy.ndarray): Shifted DFT of the image.
            magnitude_dft_original (numpy.ndarray): Magnitude spectrum.
            img_back_original (numpy.ndarray): Reconstructed image from inverse DFT.
    """
    # Compute DFT
    dft = np.fft.fft2(img_gray)
    dft_shift = np.fft.fftshift(dft)

    # Compute magnitude spectrum (log scale)
    magnitude_dft_original = np.log(np.abs(dft_shift) + 1)

    # Inverse DFT (reconstruction)
    idft_shift_original = np.fft.ifftshift(dft_shift)
    img_back_original = np.fft.ifft2(idft_shift_original)
    img_back_original = np.abs(img_back_original)

    return dft_shift, magnitude_dft_original, img_back_original


# -------------------------------------------------
#        Frequency Domain Filtering Functions
# -------------------------------------------------
def low_pass_filtering(img_gray, dft_shift, radius_low):
    """
    Apply a Low-pass filter in the frequency domain.

    Args:
        img_gray (numpy.ndarray): Grayscale image.
        dft_shift (numpy.ndarray): Shifted DFT of the image.
        radius_low (int): Radius of the low-pass mask.

    Returns:
        tuple:
            img_back (numpy.ndarray): Reconstructed low-pass filtered image.
            magnitude_dft_low_passed (numpy.ndarray): Magnitude spectrum after filtering.
    """
    low_pass_filter = low_pass_mask(dft_shift.shape, radius_low)
    low_passed = dft_shift * low_pass_filter

    # Inverse DFT
    idft_shift = np.fft.ifftshift(low_passed)
    img_back = np.fft.ifft2(idft_shift)
    img_back = np.abs(img_back)
    magnitude_dft_low_passed = np.log(np.abs(low_passed) + 1)

    return img_back, magnitude_dft_low_passed


def high_pass_filtering(img_gray, dft_shift, radius_high):
    """
    Apply a High-pass filter in the frequency domain.
    """
    high_pass_filter = high_pass_mask(dft_shift.shape, radius_high)
    high_passed = dft_shift * high_pass_filter

    # Inverse DFT
    idft_shift = np.fft.ifftshift(high_passed)
    img_back = np.fft.ifft2(idft_shift)
    img_back = np.abs(img_back)
    magnitude_dft_high_passed = np.log(np.abs(high_passed) + 1)

    return img_back, magnitude_dft_high_passed


def band_pass_filtering(img_gray, dft_shift, radius_low, radius_high):
    """
    Apply a Band-pass filter in the frequency domain.
    """
    band_pass_filter = band_pass_mask(dft_shift.shape, radius_high, radius_low)
    band_passed = dft_shift * band_pass_filter

    # Inverse DFT
    idft_shift = np.fft.ifftshift(band_passed)
    img_back = np.fft.ifft2(idft_shift)
    img_back = np.abs(img_back)
    magnitude_dft_band_passed = np.log(np.abs(band_passed) + 1)

    return img_back, magnitude_dft_band_passed


# -------------------------------------------------
#                  MASK FUNCTIONS
# -------------------------------------------------
def low_pass_mask(shape, radius):
    """
    Create a circular Low-pass mask.

    Args:
        shape (tuple): Shape of the image (rows, cols).
        radius (int): Radius of the circular mask.

    Returns:
        numpy.ndarray: Low-pass mask.
    """
    rows, cols = shape[:2]
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    return mask


def high_pass_mask(shape, radius):
    """
    Create a High-pass mask by inverting a Low-pass mask.
    """
    return 1 - low_pass_mask(shape, radius)


def band_pass_mask(shape, low_radius, high_radius):
    """
    Create a Band-pass mask as the difference between two Low-pass masks.
    """
    return low_pass_mask(shape, high_radius) - low_pass_mask(shape, low_radius)


# -------------------------------------------------
#                  HISTOGRAM FUNCTION
# -------------------------------------------------
def histogram(image):
    """
    Compute the histogram of an image.

    Args:
        image (numpy.ndarray): Grayscale image.

    Returns:
        numpy.ndarray: Histogram (256 bins).
    """
    hist = np.zeros(256)
    for value in image.flatten():
        int_val = int(value)
        if 0 <= int_val < 256:
            hist[int_val] += 1
    return hist


# -------------------------------------------------
#                  DISPLAY FUNCTION
# -------------------------------------------------
def display(img_set, img_title):
    """
    Display multiple images or histograms in a single figure.

    Args:
        img_set (list): List of images or histograms.
        img_title (list): Corresponding titles for each image.
    """
    rows = (len(img_set) + 2) // 3
    plt.figure(figsize=(15, 5 * rows))

    for i in range(len(img_set)):
        plt.subplot(rows, 3, i + 1)
        if img_set[i].ndim == 2:  # Image
            plt.imshow(img_set[i], cmap='gray')
            plt.title(img_title[i])
            plt.axis('off')
        else:  # Histogram
            plt.bar(np.arange(256), img_set[i], color='blue')
            plt.xlim([0, 256])
            plt.title(img_title[i])

    plt.tight_layout()
    plt.show(block=True)


# -------------------------------------------------
#                  ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    main()
