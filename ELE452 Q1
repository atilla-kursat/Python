import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
# Define Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2)
# Read the image
img = cv2.imread(r'box.jpeg')
inverted_image = 255 - img

horizontal_image = cv2.rotate(inverted_image, cv2.ROTATE_90_CLOCKWISE)

# Convert the image to grayscale
gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

# Calculate the vertical histogram
hist_vertical, bins_vertical = np.histogram(gray.ravel(), 256, [0, 256])

# Fit Gaussian to the vertical histogram
x_vertical = np.linspace(0, 255, 256)
popt_vertical, _ = curve_fit(gaussian, x_vertical, hist_vertical, p0=[np.max(hist_vertical), np.mean(gray), np.std(gray)])

# Transpose the grayscale image to calculate horizontal histogram
gray_transpose = cv2.cvtColor(horizontal_image, cv2.COLOR_BGR2GRAY)
gray_transpose = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)


# Calculate the horizontal histogram
hist_horizontal, bins_horizontal = np.histogram(gray_transpose.ravel(), 256, [0, 256])

# Fit Gaussian to the horizontal histogram
x_horizontal = np.linspace(0, 255, 256)
popt_horizontal, _ = curve_fit(gaussian, x_horizontal, hist_horizontal, p0=[np.max(hist_horizontal), np.mean(gray), np.std(gray)])



# Calculate peak value, standard deviation, variance, and FWHM for vertical histogram
peak_value_vertical = popt_vertical[0]
std_dev_vertical = popt_vertical[2]
variance_vertical = std_dev_vertical ** 2
half_max_vertical = peak_value_vertical / 2
left_idx_vertical = np.argmin(np.abs(hist_vertical[:int(len(hist_vertical) / 2)] - half_max_vertical))
right_idx_vertical = np.argmin(np.abs(hist_vertical[int(len(hist_vertical) / 2):] - half_max_vertical)) + int(len(hist_vertical) / 2)
fwhm_vertical = bins_vertical[right_idx_vertical] - bins_vertical[left_idx_vertical]

# Calculate peak value, standard deviation, variance, and FWHM for horizontal histogram
peak_value_horizontal = popt_horizontal[0]
std_dev_horizontal = popt_horizontal[2]
variance_horizontal = std_dev_horizontal ** 2
half_max_horizontal = peak_value_horizontal / 2
left_idx_horizontal = np.argmin(np.abs(hist_horizontal[:int(len(hist_horizontal) / 2)] - half_max_horizontal))
right_idx_horizontal = np.argmin(np.abs(hist_horizontal[int(len(hist_horizontal) / 2):] - half_max_horizontal)) + int(len(hist_horizontal) / 2)
fwhm_horizontal = bins_horizontal[right_idx_horizontal] - bins_horizontal[left_idx_horizontal]

# Plot the histograms with Gaussian fits
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(hist_vertical, label='Vertical Histogram')
plt.plot(x_vertical, gaussian(x_vertical, *popt_vertical), color='red', label='Vertical Gaussian Fit')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.title('Vertical Grayscale Histogram with Gaussian Fit\nPeak Value: {:.2f}, Std Dev: {:.2f}, Variance: {:.2f}, FWHM: {:.2f}'.format(peak_value_vertical, std_dev_vertical, variance_vertical, fwhm_vertical))
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_horizontal, label='Horizontal Histogram')
plt.plot(x_horizontal, gaussian(x_horizontal, *popt_horizontal), color='red', label='Horizontal Gaussian Fit')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.title('Horizontal Grayscale Histogram with Gaussian Fit\nPeak Value: {:.2f}, Std Dev: {:.2f}, Variance: {:.2f}, FWHM: {:.2f}'.format(peak_value_horizontal, std_dev_horizontal, variance_horizontal, fwhm_horizontal))
plt.legend()

plt.tight_layout()
plt.show()
