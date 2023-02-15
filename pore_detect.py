import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert to grayscale
img = cv2.imread('/home/skin/face_data/test0/Rgb_Cool.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray = cv2.pyrDown(gray)

# Apply median filter to remove noise and small details
median = cv2.medianBlur(gray, 5)

# Apply edge detection using Sobel operator
sobelx = cv2.Sobel(median, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(median, cv2.CV_64F, 0, 1, ksize=3)
gradient = np.sqrt(np.square(sobelx) + np.square(sobely))

# Threshold gradient magnitude to obtain binary image
threshold = 50
binary = np.zeros_like(gradient)
binary[gradient > threshold] = 255

# Apply dilation and erosion to connect and shrink edges
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Apply non-maximum suppression to detect pores
radius = 15
suppressed = np.zeros_like(closed)
for y in range(radius, closed.shape[0] - radius):
    for x in range(radius, closed.shape[1] - radius):
        if closed[y, x] == 255:
            patch = closed[y-radius:y+radius+1, x-radius:x+radius+1]
            if np.max(patch) == 255:
                suppressed[y, x] = 255

# Draw circles on original image to mark pore locations
pores = np.argwhere(suppressed == 255)
for center in pores:
    cv2.circle(img, tuple(center[::-1]), radius, (0, 0, 255), 2)

cv2.imwrite("pores.png", img)
