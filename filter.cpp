/*
Anisha Kumari Kushwaha
Sarthak Kagliwal
26th January 2024
Project 1: Video Special effects
Implemented filter effects on video based on keypress
*/

#include <opencv2/opencv.hpp>

int greyscale(cv::Mat &src, cv::Mat &dst) {

    /*
    Function to convert a color image to greyscale
    Arguments:
    - src: Source image (color)
    - dst: Destination image (greyscale)
    Returns:
    0 on success, -1 on failure
    */
    
    if (src.empty()) {
        return -1; // Return an error if source image is empty
    }
    dst = cv::Mat(src.rows, src.cols, CV_8U);
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            uchar minValue = std::min(std::min(pixel[0], pixel[1]), pixel[2]);
            dst.at<uchar>(y, x) = minValue;
        }
    }
    return 0;
}


int applySepiaTone(cv::Mat &src, cv::Mat &dst) {
    /*
    Function to apply sepia tone effect to an image
    Arguments:
    - src: Source image
    - dst: Destination image (with sepia tone)
    Returns:
    0 on success, -1 on failure
    */
    if (src.empty()) {
        return -1; // Return an error if source image is empty
    }

    // Create a destination image of the same size and type as source
    dst = src.clone();

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

            // Applying sepia matrix
            float newBlue = 0.272 * pixel[2] + 0.534 * pixel[1] + 0.131 * pixel[0];
            float newGreen = 0.349 * pixel[2] + 0.686 * pixel[1] + 0.168 * pixel[0];
            float newRed = 0.393 * pixel[2] + 0.769 * pixel[1] + 0.189 * pixel[0];

            // Clamping the values to be in the 0-255 range
            dst.at<cv::Vec3b>(y, x)[2] = std::min(255.0f, newRed);
            dst.at<cv::Vec3b>(y, x)[1] = std::min(255.0f, newGreen);
            dst.at<cv::Vec3b>(y, x)[0] = std::min(255.0f, newBlue);
        }
    }
    return 0; // Return success
}

int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    /*
    Function to apply a 5x5 blur filter to an image (version 1)
    Arguments:
    - src: Source image
    - dst: Destination image (blurred)
    Returns:
    0 on success, -1 on failure
    */
    if (src.empty()) {
        return -1;
    }

    dst = src.clone();
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    int kernelSum = 100; // Sum of kernel values

    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            // Separate channels
            int blueSum = 0, greenSum = 0, redSum = 0;

            // Apply the kernel
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);
                    blueSum += pixel[0] * kernel[ky + 2][kx + 2];
                    greenSum += pixel[1] * kernel[ky + 2][kx + 2];
                    redSum += pixel[2] * kernel[ky + 2][kx + 2];
                }
            }

            // Normalize and assign to the destination pixel
            dst.at<cv::Vec3b>(y, x)[0] = blueSum / kernelSum;
            dst.at<cv::Vec3b>(y, x)[1] = greenSum / kernelSum;
            dst.at<cv::Vec3b>(y, x)[2] = redSum / kernelSum;
        }
    }

    return 0;
}


int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    /*
    Function to apply a Seperable 1x5 blur filter to an image (version 2)
    Arguments:
    - src: Source image
    - dst: Destination image (blurred)
    Returns:
    0 on success, -1 on failure
    */

    if (src.empty()) {
        return -1;
    }

    cv::Mat temp = src.clone();
    dst = src.clone();
    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10; // Sum of 1D kernel values

    // Horizontal pass
    for (int y = 0; y < src.rows; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            int blueSum = 0, greenSum = 0, redSum = 0;
            for (int kx = -2; kx <= 2; kx++) {
                cv::Vec3b pixel = dst.at<cv::Vec3b>(y, x + kx);
                blueSum += pixel[0] * kernel[kx + 2];
                greenSum += pixel[1] * kernel[kx + 2];
                redSum += pixel[2] * kernel[kx + 2];
            }
            dst.at<cv::Vec3b>(y, x)[0] = blueSum / kernelSum;
            dst.at<cv::Vec3b>(y, x)[1] = greenSum / kernelSum;
            dst.at<cv::Vec3b>(y, x)[2] = redSum / kernelSum;
        }
    }

    // Vertical pass
    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 0; x < src.cols; x++) {
            int blueSum = 0, greenSum = 0, redSum = 0;
            for (int ky = -2; ky <= 2; ky++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x);
                blueSum += pixel[0] * kernel[ky + 2];
                greenSum += pixel[1] * kernel[ky + 2];
                redSum += pixel[2] * kernel[ky + 2];
            }
            dst.at<cv::Vec3b>(y, x)[0] = blueSum / kernelSum;
            dst.at<cv::Vec3b>(y, x)[1] = greenSum / kernelSum;
            dst.at<cv::Vec3b>(y, x)[2] = redSum / kernelSum;
        }
    }

    // dst = temp.clone();
    return 0;
}


int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    /*
    Function to apply a 3x3 Sobel filter along the X-axis to an image
    Arguments:
    - src: Source image
    - dst: Destination image (sobelX)
    Returns:
    0 on success, -1 on failure
    */

    if (src.empty() || src.type() != CV_8UC3) {
        return -1;
    }

    cv::Mat kernel = (cv::Mat_<float>(1, 3) << -1, 0, 1);
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            for (int c = 0; c < 3; c++) {
                short sobel_val = src.at<cv::Vec3b>(y, x + 1)[c] - src.at<cv::Vec3b>(y, x - 1)[c];
                dst.at<cv::Vec3s>(y, x)[c] = sobel_val;
            }
        }
    }

    return 0;
}


int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    /*
    Function to apply a 3x3 Sobel filter along the Y-axis to an image
    Arguments:
    - src: Source image
    - dst: Destination image (sobelY)
    Returns:
    0 on success, -1 on failure
    */
    if (src.empty() || src.type() != CV_8UC3) {
        return -1;
    }

    cv::Mat kernel = (cv::Mat_<float>(3, 1) << 1, 0, -1);
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            for (int c = 0; c < 3; c++) {
                short sobel_val = src.at<cv::Vec3b>(y + 1, x)[c] - src.at<cv::Vec3b>(y - 1, x)[c];
                dst.at<cv::Vec3s>(y, x)[c] = sobel_val;
            }
        }
    }

    return 0;
}


int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    /*
    Function to calculate the magnitude of gradients in an image
    Arguments:
    - sx: Image with gradients along the X-axis
    - sy: Image with gradients along the Y-axis
    - dst: Destination image (magnitude)
    Returns:
    0 on success, -1 on failure
    */
    if (sx.empty() || sy.empty() || sx.size() != sy.size()) {
        return -1;
    }

    dst = cv::Mat::zeros(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; y++) {
        for (int x = 0; x < sx.cols; x++) {
            cv::Vec3s sxPixel = sx.at<cv::Vec3s>(y, x);
            cv::Vec3s syPixel = sy.at<cv::Vec3s>(y, x);

            cv::Vec3b magnitudePixel;
            for (int c = 0; c < 3; c++) {
                // Calculate gradient magnitude per channel
                float magnitude = sqrt(sxPixel[c] * sxPixel[c] + syPixel[c] * syPixel[c]);
                magnitudePixel[c] = cv::saturate_cast<uchar>(magnitude);
            }

            dst.at<cv::Vec3b>(y, x) = magnitudePixel;
        }
    }
    return 0;
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    /*
    Function to apply a 5x5 blur filter and quantize the image to a specified number of levels
    Arguments:
    - src: Source image
    - dst: Destination image (blurred and quantized)
    - levels: Number of quantization levels
    Returns:
    0 on success, -1 on failure
    */
    if (src.empty() || levels <= 0) {
        return -1;
    }

    // Blurring the image - using a simple averaging filter for demonstration
    cv::blur(src, dst, cv::Size(5, 5));

    int bucketSize = 255 / levels;
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            for (int c = 0; c < 3; c++) {
                uchar &pixel = dst.at<cv::Vec3b>(y, x)[c];
                pixel = static_cast<int>(pixel / bucketSize) * bucketSize;
            }
        }
    }

    return 0;
}

// task 11
// Colorful Face with Greyscale Background
void colorfulFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade) {
    /*
    Function to highlight colorful faces on a greyscale background
    Arguments:
    - src: Source image
    - dst: Destination image (colorful faces on grey background)
    - faceCascade: Cascade classifier for face detection
    Returns:
    void
    */
    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, dst, cv::COLOR_GRAY2BGR);  // Convert to BGR format for overlay

    faceCascade.detectMultiScale(gray, faces);

    for (const auto &face : faces) {
        src(face).copyTo(dst(face));
    }
}

// Embossing Effect
void embossEffect(cv::Mat &src, cv::Mat &dst) {
    /*
    Function to apply an emboss effect to an image
    Arguments:
    - src: Source image
    - dst: Destination image (embossed)
    Returns:
    void
    */
    cv::Mat sobelX, sobelY;
    sobelX3x3(src, sobelX);  // Assuming sobelX3x3 is implemented
    sobelY3x3(src, sobelY);  // Assuming sobelY3x3 is implemented

    cv::Mat emboss = cv::Mat::zeros(src.size(), CV_16SC3);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3s xPixel = sobelX.at<cv::Vec3s>(y, x);
            cv::Vec3s yPixel = sobelY.at<cv::Vec3s>(y, x);
            emboss.at<cv::Vec3s>(y, x) = xPixel + yPixel;
        }
    }
    cv::convertScaleAbs(emboss, dst);
}

// Blur Outside of Faces
void blurOutsideFaces(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade) {
    /*
    Function to blur the areas outside of detected faces in an image
    Arguments:
    - src: Source image
    - dst: Destination image (blurred outside faces)
    - faceCascade: Cascade classifier for face detection
    Returns:
    void
    */
    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    faceCascade.detectMultiScale(gray, faces);

    // Initialize the mask as all black
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);

    // Set the regions containing faces to white
    for (const auto &face : faces) {
        mask(face).setTo(cv::Scalar(255));
    }
   
    // Blur the entire source image
    cv::GaussianBlur(src, dst, cv::Size(21, 21), 0, 0);

    // Copy only the faces from the original image to the blurred image
    src.copyTo(dst, mask);
}




// Extension


// extension 1
void cartoonEffect(cv::Mat &src, cv::Mat &dst) {
    /*
    Extension 1: Function to apply a cartoon effect to an image
    Arguments:
    - src: Source image
    - dst: Destination image (cartoon effect)
    Returns:
    void
    */
    // Apply a median blur to reduce image noise
    cv::Mat blurred;
    cv::medianBlur(src, blurred, 7);

    // Convert to edges and invert the image for a sketch-like effect
    cv::Mat edges;
    cv::Canny(blurred, edges, 25, 50, 3);
    cv::bitwise_not(edges, edges);

    // Reduce the color palette
    cv::Mat reducedColor;
    cv::convertScaleAbs(src, reducedColor, 1/2.0, 0);
    cv::convertScaleAbs(reducedColor, reducedColor, 2, 0);

    // Increase the darkness of the edges (the border)
    cv::multiply(edges, cv::Scalar(10, 10, 10), edges);

    // Combine edges and color
    cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
    dst = reducedColor & edges;
}

// extension 2
// adjust brightness
void adjustBrightness(cv::Mat &src, cv::Mat &dst, int brightness) {
    /*
    Extension 2: Function to adjust the brightness of an image
    Arguments:
    - src: Source image
    - dst: Destination image (brightness adjusted)
    - brightness: Adjustment value
    Returns:
    void
    */
    // Ensure frame is not empty
    if (src.empty()) {
        return;
    }

    // Clone the source image to the destination
    src.copyTo(dst);

    // Calculate alpha and beta values based on brightness input
    double alpha = 1.0 + brightness / 255.0;  // Scale brightness from [-255, 255] to [0, 2]
    double beta = -128 + brightness;        // Shift brightness within [-128, 127]

    // Iterate through each pixel and adjust brightness
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < src.channels(); c++) {
                int pixelValue = src.at<cv::Vec3b>(y, x)[c];
                // Apply brightness adjustment formula
                int newPixelValue = cv::saturate_cast<uchar>(alpha * pixelValue + beta);
                dst.at<cv::Vec3b>(y, x)[c] = newPixelValue;
            }
        }
    }
}

//extension 3
int customNegative(cv::Mat &src, cv::Mat &dst) {
    /*
    Extension 3: Function to create a custom negative of an image
    Arguments:
    - src: Source image
    - dst: Destination image (custom negative)
    Returns:
    0 on success, -1 on failure
    */
    // Ensure src is not empty
    if (src.empty()) {
        return -1;
    }

    // Convert to greyscale using a custom method
    cv::Mat temp(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Example transformation: invert the red channel and assign to all channels
            uchar pixel = 255 - src.at<cv::Vec3b>(i, j)[2];
            temp.at<uchar>(i, j) = pixel;
        }
    }
    cv::cvtColor(temp, dst, cv::COLOR_GRAY2BGR);
    return 0;
}

//extension 4
void invertColorsWithTint(cv::Mat &src, cv::Mat &dst, const cv::Vec3b &tint) {
    /*
    Extension 4: Function to invert colors with a tint
    Arguments:
    - src: Source image
    - dst: Destination image (inverted with tint)
    - tint: Tint color to add during inversion
    Returns:
    - void
    */
    dst = src.clone();
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            // Invert each color channel
            dst.at<cv::Vec3b>(y, x)[0] = 255 - dst.at<cv::Vec3b>(y, x)[0] + tint[0];
            dst.at<cv::Vec3b>(y, x)[1] = 255 - dst.at<cv::Vec3b>(y, x)[1] + tint[1];
            dst.at<cv::Vec3b>(y, x)[2] = 255 - dst.at<cv::Vec3b>(y, x)[2] + tint[2];

            // Ensure pixel values are within [0, 255]
            for (int c = 0; c < 3; c++) {
                if (dst.at<cv::Vec3b>(y, x)[c] > 255) {
                    dst.at<cv::Vec3b>(y, x)[c] = 255;
                }
            }
        }
    }
}

// extension 5
void customColorFilter(cv::Mat &src, cv::Mat &dst, const cv::Vec3f &colorMultiplier) {
    /*
    Extension 5: Function to apply a custom color filter
    Arguments:
    - src: Source image
    - dst: Destination image (filtered)
    - colorMultiplier: Multiplier for each color channel
    Returns:
    - void
    */
    dst = src.clone();
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);
            pixel[0] = cv::saturate_cast<uchar>(pixel[0] * colorMultiplier[0]); // Blue
            pixel[1] = cv::saturate_cast<uchar>(pixel[1] * colorMultiplier[1]); // Green
            pixel[2] = cv::saturate_cast<uchar>(pixel[2] * colorMultiplier[2]); // Red
        }
    }
}

//extension 6
void artisticBrushStrokeEffect(cv::Mat &src, cv::Mat &dst, int strokeSize) {
    /*
    Extension 6: Function to create an artistic brush stroke effect
    Arguments:
    - src: Source image
    - dst: Destination image (brush stroke effect)
    - strokeSize: Size of the brush stroke
    Return:
    - void
    */
    dst = src.clone();
    cv::Mat blurredSrc;
    cv::GaussianBlur(src, blurredSrc, cv::Size(3, 3), 0, 0);

    for (int y = 0; y < src.rows; y += strokeSize) {
        for (int x = 0; x < src.cols; x += strokeSize) {
            // Compute dominant gradient direction in the patch
            float avgDx = 0.0, avgDy = 0.0;
            for (int i = y; i < y + strokeSize && i < src.rows - 1; ++i) {
                for (int j = x; j < x + strokeSize && j < src.cols - 1; ++j) {
                    cv::Vec3b pixCurr = blurredSrc.at<cv::Vec3b>(i, j);
                    cv::Vec3b pixRight = blurredSrc.at<cv::Vec3b>(i, j + 1);
                    cv::Vec3b pixDown = blurredSrc.at<cv::Vec3b>(i + 1, j);

                    avgDx += pixRight[1] - pixCurr[1];
                    avgDy += pixDown[1] - pixCurr[1];
                }
            }

            avgDx /= strokeSize * strokeSize;
            avgDy /= strokeSize * strokeSize;

            // Apply brush stroke effect
            for (int i = y; i < y + strokeSize && i < src.rows; ++i) {
                for (int j = x; j < x + strokeSize && j < src.cols; ++j) {
                    int shiftX = avgDx > 0 ? strokeSize / 4 : -strokeSize / 4;
                    int shiftY = avgDy > 0 ? strokeSize / 4 : -strokeSize / 4;

                    int srcY = std::min(std::max(i + shiftY, 0), src.rows - 1);
                    int srcX = std::min(std::max(j + shiftX, 0), src.cols - 1);

                    dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(srcY, srcX);
                }
            }
        }
    }
}


