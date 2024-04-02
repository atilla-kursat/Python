import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread(r'blackbox1.jpeg')
inverted_image = 255 - img

# Convert the image to grayscale
gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
# Extract a horizontal line profile from the middle of the image
horizontal_profile = gray[gray.shape[0] // 2, :]

# Plot the pixel intensities along the horizontal direction
plt.figure(figsize=(10, 6))
plt.plot(horizontal_profile, label='Vertical Line Profile', color='blue')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')
plt.title('Pixel Intensities along Vertical Direction')
plt.legend()
plt.grid(True)
plt.show()

# Zoom into the edges to observe the blurring effect
plt.figure(figsize=(10, 6))
plt.plot(horizontal_profile, label='Vertical Line Profile', color='blue')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')
plt.title('Pixel Intensities along Vertical Direction (Zoomed)')
plt.xlim(200, 300)  # Adjust the limits to zoom into the edges
plt.ylim(0, 255)     # Set the intensity range
plt.legend()
plt.grid(True)
plt.show()

# Fit a first-degree polynomial (linear fit) to the pixel intensities
x = np.arange(len(horizontal_profile))
coefficients_linear = np.polyfit(x, horizontal_profile, 1)
fit_linear = np.poly1d(coefficients_linear)
print(fit_linear)

# Fit a second-degree polynomial (quadratic fit) to the pixel intensities
coefficients_quadratic = np.polyfit(x, horizontal_profile, 2)
fit_quadratic = np.poly1d(coefficients_quadratic)
print(fit_quadratic)
# Plot the original data and polynomial fits
plt.figure(figsize=(10, 6))
plt.plot(horizontal_profile, label='Vertical Line Profile', color='blue')
plt.plot(x, fit_linear(x), label='Linear Fit', linestyle='--', color='red')
plt.plot(x, fit_quadratic(x), label='Quadratic Fit', linestyle='--', color='green')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')
plt.title('Pixel Intensities along Vertical Direction with Polynomial Fits')
plt.legend()
plt.grid(True)
plt.show()
