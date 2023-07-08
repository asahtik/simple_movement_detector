#include <iostream>
#include "optical_flow.hpp"

int main(int argc, char** argv) {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Camera not found" << std::endl;
        return -1;
    }
    cv::Mat frame1;
    bool initialised = false;
    while (true) {
        cv::Mat frame2;
        cap >> frame2;
        if (!initialised) {
            frame1 = frame2;
            initialised = true;
            continue;
        }
        // TODO
    }
}
