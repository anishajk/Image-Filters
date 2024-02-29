/*
Anisha Kumari Kushwaha
Sarthak Kagliwal
26th January 2024
Project 1: Video Special effects
Opens a video channel, creates a window and applies various effects on keypress
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "faceDetect.h"
#include <sys/time.h>
#include "filter.h"


// Define an enumeration for different filter modes
enum FilterMode {
    MODE_NORMAL,
    MODE_GREYSCALE,
    MODE_CUSTOM_GREYSCALE,
    MODE_SEPIA,
    MODE_BLUR_1,
    MODE_BLUR_2,
    MODE_SOBEL_X,
    MODE_SOBEL_Y,
    MODE_MAGNITUDE,
    MODE_BLUR_QUANTIZE,
    MODE_FACE_DETECT,
    MODE_COLOR_FACE_GREYBG,
    MODE_EMBOSS,
    MODE_BLUR_3,
    MODE_CARTOON,
    MODE_ADJUST_BRIGHTNESS,
    MODE_CUSTOM_NEGATIVE,
    MODE_INVERT_COLORS_TINT,
    MODE_COLOR,
    MODE_ARTISTIC_BRUSH_STROKE,
};


// Function to save the current frame with an optional comment displayed on the image
void saveCurrentFrame(const cv::Mat &frame, const std::string &filename, const std::string &comment = "") {
    if (!frame.empty()) {
        cv::Mat frameWithComment = frame.clone();

        // If a comment is provided, draw it on the image
        if (!comment.empty()) {
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.8;
            int thickness = 2;
            cv::Point textOrg(30, 50);  // Adjust the position of the comment as needed
            cv::putText(frameWithComment, comment, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
        }

        cv::imwrite(filename, frameWithComment);
        std::cout << "Frame saved as " << filename << std::endl;
    } else {
        std::cerr << "Error: Empty frame, not saved." << std::endl;
    }
}

// returns a double which gives time in seconds
double getTime() {
  struct timeval cur;

  gettimeofday( &cur, NULL );
  return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}
  

int main(int argc, char *argv[]) {
    cv::VideoCapture cap(0); // Open the video camera
    cv::CascadeClassifier faceCascade;
    std::string cascadePath = FACE_CASCADE_FILE;
    faceCascade.load(cascadePath);
    int brightness = 0;
    cv::Mat processedFrame;
    cv::Mat sobelFrameX;
    cv::Mat sobelFrameY;
    cv::Mat grey;
    std::vector<cv::Rect> faces;
    cv::Rect last(0, 0, 0, 0);
    cv::VideoWriter videoWriter;
    bool isRecording = false;
    cv::Vec3f colorMultiplier(1.0, 0.5, 0.5);


    std::string comment;  // Variable to store the user's comment for saving the video frame
    std::string caption;  // Variable to store the user's caption for saving images or video frames


    const int Ntimes = 10;
     // set up the timing for version 1
    double startTime;
    // end the timing
    double endTime;
    // compute the time per image
    double difference;

    // Check if the video camera is successfully opened
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Unable to open the camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::namedWindow("Video", 1); // Create a window

    FilterMode currentMode = MODE_NORMAL;

    while (true) {
        cap >> frame; // Get a new frame from the camera
        if (frame.empty()) {
            std::cerr << "ERROR: Frame is empty" << std::endl;
            break;
        }

        char key = (char)cv::waitKey(10);
        // Switch statement to handle key presses
        switch (key) {
            case 'q': return 0;
            case 's': cv::imwrite("saved_frame.jpg", frame); break;
            case 'g': currentMode = MODE_GREYSCALE; break;
            case 'h': currentMode = MODE_CUSTOM_GREYSCALE; break;
            case 'p': currentMode = MODE_SEPIA; break;
            case '1': currentMode = MODE_BLUR_1; break;
            case '2': currentMode = MODE_BLUR_2; break;
            case 'x': currentMode = MODE_SOBEL_X; break;
            case 'y': currentMode = MODE_SOBEL_Y; break;
            case 'm': currentMode = MODE_MAGNITUDE; break;
            case '3': currentMode = MODE_BLUR_QUANTIZE; break;
            case 'f': currentMode = MODE_FACE_DETECT; break;
            case 'd': currentMode = MODE_COLOR_FACE_GREYBG; break;
            case 'e': currentMode = MODE_EMBOSS; break;
            case '4': currentMode = MODE_BLUR_3; break;
            case 'j': currentMode = MODE_CARTOON; break;
            case 'i': currentMode = MODE_ADJUST_BRIGHTNESS; break;
            case 'n': currentMode = MODE_CUSTOM_NEGATIVE; break;
            case 'w': currentMode = MODE_INVERT_COLORS_TINT; break;
            case 'z': currentMode = MODE_COLOR; break;
            case 'v': currentMode = MODE_ARTISTIC_BRUSH_STROKE; break;
            case '.': 
                // Save the current processed frame as an image with an optional caption
                std::cout << "Enter a caption for saving the image (press Enter to skip): ";
                std::getline(std::cin, caption);
                saveCurrentFrame(processedFrame, "saved_frame.jpg", caption);
                caption.clear();  // Clear the caption for the next image
                break;
            case 'r':
                if (!isRecording) {
                    // Start recording
                    std::cout << "Recording started...";
                    videoWriter.open("recorded_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, frame.size(), true);
                    isRecording = videoWriter.isOpened();
                    // Prompt the user for a comment
                    std::cout << "Enter a comment for saving the video (press Enter to skip): ";
                    std::getline(std::cin, comment);
                } else {
                    // Stop recording
                    std::cout << "Recording Ended!";
                    videoWriter.release();
                    isRecording = false;
                }
                break;
            default: break;
        }

        // Switch statement to apply different image processing modes based on the current mode
        switch (currentMode) {
            case MODE_GREYSCALE:    
                // Convert the frame to greyscale and then back to BGR using cvtColor
                cv::cvtColor(frame, processedFrame, cv::COLOR_BGR2GRAY);
                cv::cvtColor(processedFrame, processedFrame, cv::COLOR_GRAY2BGR);
                break;
            case MODE_CUSTOM_GREYSCALE:
                greyscale(frame, processedFrame);
                break;
            case MODE_SEPIA:
                applySepiaTone(frame, processedFrame);
                break;
            case MODE_BLUR_1:
                 // set up the timing for version 1
                startTime = getTime();

                blur5x5_1(frame, processedFrame);

                endTime = getTime();
                // compute the time per image
                difference = (endTime - startTime) / Ntimes;
                // print the results
                printf("Time per image to blur_1: %.4lf seconds\n", difference );
                break;
            case MODE_BLUR_2:
                // set up the timing for version 2
                startTime = getTime();

                blur5x5_2(frame, processedFrame);

                endTime = getTime();
                // compute the time per image
                difference = (endTime - startTime) / Ntimes;
                // print the results
                printf("Time per image to blur_2: %.4lf seconds\n", difference );
                break;
            case MODE_SOBEL_X:
                sobelX3x3(frame, sobelFrameX);
                cv::convertScaleAbs(sobelFrameX, processedFrame);
                break;
            case MODE_SOBEL_Y:
                sobelY3x3(frame, sobelFrameY);
                cv::convertScaleAbs(sobelFrameY, processedFrame);
                break;
            case MODE_MAGNITUDE:
                sobelX3x3(frame, sobelFrameX);
                sobelY3x3(frame, sobelFrameY);
                magnitude(sobelFrameX, sobelFrameY, processedFrame);
                break;
            case MODE_BLUR_QUANTIZE:
                blurQuantize(frame, processedFrame,10);
                break;
            case MODE_FACE_DETECT:
                // Detect faces and draw boxes around them
                cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
                detectFaces(grey, faces);
                drawBoxes(frame, faces);
                if(faces.size() > 0) {
                    // Update the last face position for smoothing
                    last.x = (faces[0].x + last.x)/2;
                    last.y = (faces[0].y + last.y)/2;
                    last.width = (faces[0].width + last.width)/2;
                    last.height = (faces[0].height + last.height)/2;
                }
                processedFrame = frame;
                break;
            case MODE_COLOR_FACE_GREYBG:
                colorfulFace(frame, processedFrame, faceCascade);
                break;
            case MODE_EMBOSS:
                embossEffect(frame, processedFrame);
                break;
            case MODE_BLUR_3:
                blurOutsideFaces(frame, processedFrame, faceCascade);
                break;
            case MODE_CARTOON:
                cartoonEffect(frame, processedFrame);
                break;
            case MODE_ADJUST_BRIGHTNESS:
                // Adjust brightness with specific keys (e.g., '+' to increase, '-' to decrease)
                if (key == '+') { brightness += 10; } // shift + '+'
                if (key == '-') { brightness -= 10; } // -
                
                adjustBrightness(frame, processedFrame, brightness);
                break;
            case MODE_CUSTOM_NEGATIVE:
                customNegative(frame, processedFrame);
                break;
            case MODE_INVERT_COLORS_TINT:
                invertColorsWithTint(frame, processedFrame, cv::Vec3b(10, 50, 70)); 
                break;
            case MODE_COLOR:
                customColorFilter(frame, processedFrame, colorMultiplier);
                break;
            case MODE_ARTISTIC_BRUSH_STROKE:
                artisticBrushStrokeEffect(frame, processedFrame, 10); 
                break;
            default:
                processedFrame = frame;
                break;
        }
        if (isRecording) {
            // Display comment on the recorded video
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.8;
            int thickness = 2;
            cv::Point textOrg(30, 50);  // Adjust the position of the comment as needed
            cv::putText(processedFrame, comment, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);

            videoWriter.write(processedFrame);
        }
        cv::imshow("Video", processedFrame);
    }

    cap.release();
    if (videoWriter.isOpened()) {
        videoWriter.release();
    }
    cv::destroyAllWindows();
    return 0;
}
