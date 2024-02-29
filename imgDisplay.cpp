/*
Anisha Kumari Kushwaha
Sarthak Kagliwal
26th January 2024
Project 1: Video Special effects
Reads and display imagin a window
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    Mat image = imread("src/img/download.jpeg", IMREAD_COLOR);
    // Check if the image is successfully loaded
    if(image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Create a window named "Display window" with automatic size adjustment
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);

    // Wait for a key press indefinitely (press 'q' to exit, 's' to save)
    while (true) {
        char c = (char)waitKey(25);
        if (c == 'q') {
            break;
        } else if (c == 's') {
            // Save the current image with a new name
            imwrite("saved_image.jpg", image);
            cout << "Image saved as saved_image.jpg" << endl;
        }
    }

    return 0;
}
