import cv2
import numpy as np

# Linear contrast stretching
def linear_mapping(img):
    img_min = np.min(img)
    img_max = np.max(img)
    stretched = ((img - img_min) / (img_max - img_min)) * 255
    return stretched.astype(np.uint8)

# Nonlinear: Histogram Equalization
def nonlinear_mapping(img):
    return cv2.equalizeHist(img)

# Create histogram image
def create_histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    hist_img = np.ones((100, 256), dtype=np.uint8) * 255
    for x, y in enumerate(hist):
        cv2.line(hist_img, (x, 100), (x, 100 - int(y * 100)), 0)
    return cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fixed width
    frame = cv2.resize(frame, (256, 256))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply techniques
    linear = linear_mapping(gray)
    nonlinear = nonlinear_mapping(gray)

    # Histograms
    orig_hist = create_histogram(gray)
    linear_hist = create_histogram(linear)
    nonlinear_hist = create_histogram(nonlinear)

    # Resize histograms to match image height
    orig_hist = cv2.resize(orig_hist, (256, 256))
    linear_hist = cv2.resize(linear_hist, (256, 256))
    nonlinear_hist = cv2.resize(nonlinear_hist, (256, 256))

    # Convert gray to BGR for display
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    linear_bgr = cv2.cvtColor(linear, cv2.COLOR_GRAY2BGR)
    nonlinear_bgr = cv2.cvtColor(nonlinear, cv2.COLOR_GRAY2BGR)

    # Stack images
    top_row = np.hstack((gray_bgr, linear_bgr, nonlinear_bgr))
    bottom_row = np.hstack((orig_hist, linear_hist, nonlinear_hist))
    combined = np.vstack((top_row, bottom_row))

    # Show frame
    cv2.imshow("Original | Linear | Nonlinear + Histograms", combined)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
