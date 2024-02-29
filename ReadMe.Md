# Image Processing Project with OpenCV in C++
### Anisha Kumari Kushwaha
### Sarthak Kagliwal

## Overview

This C++ project utilizes the OpenCV library for image processing, implemented in Visual Studio. The program provides a real-time video processing experience with various filters and effects. Users can interact with the application using specific keypresses to switch between different image processing modes and functionalities.

### OS and IDE
- **Anisha Kumari Kushwaha:** Ubuntu 22.04 and Visual Studio Code
- **Sarthak Kagliwal:** macOS and Visual Studio Code

## Key Features

- **Quit (`q`):** Exits the application.
- **Save Frame (`s`):** Saves the current frame as an image named "saved_frame.jpg".
- **Grayscale Filter (`g`):** Converts the video stream to grayscale.
- **Custom Grayscale Filter (`c`):** Applies a custom grayscale transformation.
- **Sepia Tone Filter (`p`):** Gives a sepia tone effect to the video.
- **Blur 1 (`1`):** Applies a 5x5 blur filter with a predefined kernel.
- **Blur 2 (`2`):** Applies seperable of 1x5 blur filter.
- **Sobel X (`x`):** Applies Sobel edge detection in the X direction.
- **Sobel Y (`y`):** Applies Sobel edge detection in the Y direction.
- **Magnitude (`m`):** Computes the magnitude of Sobel X and Sobel Y.
- **Blur Quantize (`3`):** Blurs the image and quantizes colors to a specified number of levels.
- **Face Detection (`f`):** Detects faces in the video stream.
- **Colorful Face with Greyscale Background (`d`):** Highlights faces with color while keeping the background greyscale.
- **Emboss Effect (`e`):** Applies an embossing effect to the video.
- **Blur Outside Faces (`4`):** Blurs the background outside detected faces.
- **Cartoon Effect (`j`):** Creates a cartoon-like effect on the video.
- **Adjust Brightness (`i`):** Adjusts the brightness of the video with `+` and `-` keys.
- **Min Grayscale (`<`):** Converts each pixel to its minimum value among RGB channels.
- **Save Frame (`.`):** Saves the current frame as an image.
- **Toggle Video Recording (`r`):** Starts or stops recording the processed video. Recorded video is saved as "recorded_video.mp4".
- **Invert Pixel with color tint (`w`):** This function inverts the color of each pixel and adds a specified color tint.
- **Custom Color (`z`):** Changes the image's color tone by manipulating the RGB channels based on provided multipliers.
- **Artistic Brush-Stroke (`v`):** Mimic's the appearance of a painting by clustering pixels into brush-like strokes based on their color similarity.


## How to Use

- Download and set up the OpenCV C++ library in Visual Studio.
- Build and run the project.
- Use the specified keys mentioned above to interact with the application and explore various image processing effects.

## Notes

- Make sure to have the OpenCV library properly configured in your development environment.
- The project utilizes a cascade classifier (provided for this task) for face detection, and the cascade file path is defined as `FACE_CASCADE_FILE`.
- [Recording Link](https://drive.google.com/drive/folders/1pMjAsLgHYIlJt-w-nSaX6fvjN6-qK5XU?usp=sharing)
