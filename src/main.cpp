#include <iostream>
#include "optical_flow.hpp"


flow_params fp = {
    {1.0, 1.0},
    {25},
    {10, 0.1},
    true,
    {10, 5, 0.04, 1e-5}
};

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
        cv::Mat u, v;
        calculate_flow(frame1, frame2, fp, u, v);
        show_flow(frame1, u, v);

        frame1 = frame2;
    }
}
