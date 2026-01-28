import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('/Users/akhi/Desktop/DIP/images/HVD.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found. Make sure the file path is correct.")

plt.figure(figsize=(6,6))
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# -----------------------
# 1. DCT (Discrete Cosine Transform)
# -----------------------
def apply_dct(image):
    # Convert to float32
    img_float = np.float32(image) / 255.0
    # Apply 2D DCT
    dct_img = cv2.dct(img_float)
    # Apply inverse DCT
    idct_img = cv2.idct(dct_img)
    return dct_img, idct_img

dct_img, idct_img = apply_dct(img)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("DCT Coefficients")
plt.imshow(np.log(abs(dct_img)+1), cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Reconstructed from DCT")
plt.imshow(np.clip(idct_img,0,1), cmap='gray')
plt.axis('off')
plt.show()

# -----------------------
# 2. DWT (Discrete Wavelet Transform)
# -----------------------
def apply_dwt(image, wavelet='haar'):
    coeffs2 = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

LL, LH, HL, HH = apply_dwt(img)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.title("LL (Approximation)")
plt.imshow(LL, cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("LH (Horizontal)")
plt.imshow(LH, cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("HL (Vertical)")
plt.imshow(HL, cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title("HH (Diagonal)")
plt.imshow(HH, cmap='gray')
plt.axis('off')
plt.show()

# -----------------------
# 3. DFT (Discrete Fourier Transform)
# -----------------------
def apply_dft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)   # Shift zero freq to center
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    # Reconstruct image using inverse DFT
    idft_img = np.fft.ifft2(np.fft.ifftshift(dft_shift)).real
    return magnitude_spectrum, idft_img

magnitude_spectrum, idft_img = apply_dft(img)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("DFT Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Reconstructed from DFT")
plt.imshow(np.clip(idft_img,0,255), cmap='gray')
plt.axis('off')
plt.show()
