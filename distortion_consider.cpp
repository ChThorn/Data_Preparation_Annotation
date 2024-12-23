#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;


int main()
{
    // Load Image
    cv::Mat image = cv::imread("ant.jpg");
    
    if(image.empty())
    {
        cout<<"Error: Cold not open or find image.\n"<<endl;
        return -1;
    }
    
    // Display the original image
    cv::imshow("Original Image",image);

    // Define translation matrix: tx = 50, ty = 30
    cv::Mat translationMatrix = (cv::Mat_<double>(2,3) <<1,0,50,0,1,30);
    cv::Mat translatedImage;
    cv::warpAffine(image, translatedImage, translationMatrix, image.size());

    // Display the translated image
    cv::imshow("Translated Image", translatedImage);

    // Define rotation matrix: rotate 45 degrees around the center
    cv::Point2f center(image.cols/2.0, image.rows/2.0);
    double angle = 45.0;
    double scale = 1.0;
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, scale);
    cv::Mat rotatedImage;
    cv::warpAffine(image, rotatedImage, rotationMatrix, image.size());

    // Display the rotated image
    cv::imshow("Rotated Image", rotatedImage);

    // Wait until the user presses a key
    cv::waitKey(0);
    
    return 0;
}