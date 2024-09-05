# Lane Detection in Day and Night using OpenCV and Python

## Overview

This project implements lane detection using Python and OpenCV, aiming to detect lane markings in both day and night conditions. The system processes video frames or images to highlight the lanes, providing a useful tool for autonomous driving systems and advanced driver assistance.

## Features

- **Day and Night Detection**: The algorithm works in both day and night scenarios by adjusting image preprocessing techniques such as contrast adjustment, edge detection, and thresholding.
- **Real-Time Performance**: Efficient processing of frames from video streams (e.g., dashcam footage).
- **Lane Highlighting**: Lanes are highlighted using color overlays for easy visualization.
- **Edge Detection**: Canny edge detection is used for identifying lane boundaries.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib (optional, for visualization)

### Installation

1. Clone the repository or download the source code.
   ```bash
   git clone https://github.com/yourusername/lane-detection-day-night.git
