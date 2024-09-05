import matplotlib.pylab as plt 
import cv2 
import numpy as np 
 
def region_of_interest(img, vertices): 
    mask = np.zeros_like(img) 
    match_mask_color = 255 
    cv2.fillPoly(mask, vertices, match_mask_color) 
    masked_image = cv2.bitwise_and(img, mask) 
    return masked_image 
 
def draw_the_lines(img, lines): 
    img = np.copy(img) 
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), 
dtype=np.uint8) 
 
    left_lines = [] 
    right_lines = [] 
 
    if lines is not None: 
        for line in lines: 
            for x1, y1, x2, y2 in line: 
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0 
                 
                # Separate left and right lines based on the slope 
                if 0.3 < slope < 1.7:  # Right lane (positive slope) 
                    right_lines.append((x1, y1, x2, y2)) 
                elif -1.7 < slope < -0.3:  # Left lane (negative slope) 
                    left_lines.append((x1, y1, x2, y2)) 
 
    # Draw the left lane line 
    if left_lines: 
        left_line = np.mean(left_lines, axis=0).astype(int)  # Average position for stability 
        x1, y1, x2, y2 = left_line 
        y1_extended = img.shape[0] 
        y2_extended = int(img.shape[0] * 0.6) 
        slope = (y2 - y1) / (x2 - x1) 
        x1_extended = int(x1 - (y1 - y1_extended) / slope) 
        x2_extended = int(x2 - (y2 - y2_extended) / slope) 
        cv2.line(blank_image, (x1_extended, y1_extended), (x2_extended, 
y2_extended), (0, 255, 0), thickness=10) 
 
    # Draw the right lane line 
    if right_lines: 
        right_line = np.mean(right_lines, axis=0).astype(int)  # Average position for stability 
        x1, y1, x2, y2 = right_line 
        y1_extended = img.shape[0] 
        y2_extended = int(img.shape[0] * 0.6) 
        slope = (y2 - y1) / (x2 - x1) 
        x1_extended = int(x1 - (y1 - y1_extended) / slope) 
        x2_extended = int(x2 - (y2 - y2_extended) / slope) 
        cv2.line(blank_image, (x1_extended, y1_extended), (x2_extended, 
y2_extended), (0, 255, 0), thickness=10) 
 
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0) 
    return img 
 
def process(image): 
    height = image.shape[0] 
    width = image.shape[1] 
    region_of_interest_vertices = [ 
        (0, height), 
        (width/2 - 50, height/2 + 50), 
        (width/2 + 50, height/2 + 50), 
        (width, height) 
    ] 
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    canny_image = cv2.Canny(gray_image, 50, 150) 
    cropped_image = region_of_interest(canny_image, 
np.array([region_of_interest_vertices], np.int32),) 
 
    lines = cv2.HoughLinesP( 
        cropped_image, 
        rho=2, 
        theta=np.pi/180, 
        threshold=100, 
        lines=np.array([]), 
        minLineLength=50, 
        maxLineGap=150 
    ) 
 
    image_with_lines = draw_the_lines(image, lines) 
    return image_with_lines 
 
cap = cv2.VideoCapture('test.mp4') 
 
plt.ion()  # Enable interactive mode 
fig, ax = plt.subplots() 
 
while cap.isOpened(): 
    ret, frame = cap.read() 
    if not ret: 
        print("End of video file or failed to grab frame.") 
        break 
     
    frame = process(frame) 
    ax.clear()  # Clear the previous frame 
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
    plt.pause(0.01) 
 
    if plt.waitforbuttonpress(0.01): 
        break 
 
cap.release() 
plt.close() 