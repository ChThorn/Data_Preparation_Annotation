#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("result.jpg");
    if(img.empty()) {
        std::cout << "Error: Could not read the image file 'result.jpg'" << std::endl;
        return -1;
    }
    
    cv::namedWindow("YOLOv5 Result", cv::WINDOW_NORMAL);
    cv::imshow("YOLOv5 Result", img);
    cv::waitKey(0);
    
    return 0;
}
