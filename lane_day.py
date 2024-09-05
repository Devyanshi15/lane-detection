import matplotlib.pylab as plt 
import cv2 
import numpy as np 
 
def region_of_interest(img, vertices): 
    mask = np.zeros_like(img) 
    match_mask_color = 255  # Only one channel needed for grayscale images 
    cv2.fillPoly(mask, vertices, match_mask_color) 
    masked_image = cv2.bitwise_and(img, mask) 
    return masked_image 
 
def draw_the_lines(img, lines): 
    img = np.copy(img) 
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), 
dtype=np.uint8) 
 
    for line in lines: 
        for x1, y1, x2, y2 in line: 
            # Draw thicker lines and slightly brighter green color 
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 
thickness=9) 
 
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0) 
    return img 
 
# Read the image and process it 
image = cv2.imread('road1.jpg') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
height = image.shape[0] 
width = image.shape[1] 
 
# Define region of interest 
region_of_interest_vertices = [ 
    (0, height), 
    (width / 2, height / 2), 
    (width, height) 
] 
 
# Convert to grayscale and apply Canny edge detection 
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
canny_image = cv2.Canny(gray_image, 100, 200) 
 
# Apply region of interest mask 
cropped_image = region_of_interest( 
    canny_image, 
    np.array([region_of_interest_vertices], np.int32), 
) 
 
# Detect lines using Hough Transform 
lines = cv2.HoughLinesP( 
cropped_image, 
rho=6, 
theta=np.pi / 180, 
threshold=160, 
lines=np.array([]), 
minLineLength=40, 
maxLineGap=25 
) 
# Draw lines on the original image 
image_with_lines = draw_the_lines(image, lines) 
# Display the result 
plt.imshow(image_with_lines) 
plt.show() 