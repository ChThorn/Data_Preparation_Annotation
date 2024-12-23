// ----Display Image using opencv----
// #include <opencv2/core.hpp>
// #include <opencv2/highgui.hpp>
// #include <iostream>

// using namespace cv;

// int main(int argc, char** argv)
// {
//     Mat image = imread("000091.jpg", IMREAD_COLOR);
//     if(image.empty())
//     {
//         std::cout<<"Error: the image has been incorrectly loaded."<<std::endl;
//         return 0;
//     }

//     namedWindow("My first OpenCV window");
//     imshow("My first OpenCV window", image);
//     waitKey(0);
//     return 0;
// }
// ------------------

// ----Load image from camera---
#include "opencv4/opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "iostream"

#define CV_8U 1
#define CV_MAKETYPE(depth, cn) (CV_MAT_DEPTH)

int R = 100, C = 200;
Mat m1;
m1.create(R,C,CV_8U)

int main(int, char**)
{
    cv::VideoCapture camera(0);
    if(!camera.isOpened())
    {
        std::cerr<<"ERROR: Could not open camera"<<std::endl;
        return 1;
    }
    cv::namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
    cv::Mat frame;
    camera>>frame;

    while(1)
    {
        cv::imshow("Webcam", frame);
        if(cv::waitKey(10)>=0)
        break;
    }

}