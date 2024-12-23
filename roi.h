#ifndef ROI_H
#define ROI_H

#include <opencv2/opencv.hpp>

class ROIBox {
public:
    // Constructors
    ROIBox();
    ROIBox(int x, int y, int width, int height);
    ROIBox(const cv::Rect& rect);

    // Set ROI position and size
    void setROI(int x, int y, int width, int height);
    void setROI(const cv::Rect& rect);

    // Get ROI information
    cv::Rect getROI() const { return roi; }
    cv::Mat extractROI(const cv::Mat& frame) const;

    // Validation and clipping
    bool isWithinFrame(const cv::Mat& frame) const;
    cv::Rect clipRectToROI(const cv::Rect& rect) const;

    // Draw ROI on frame
    void draw(cv::Mat& frame, const cv::Scalar& color = cv::Scalar(255, 0, 0), 
             int thickness = 2) const;

private:
    cv::Rect roi;
    void validateROI();
};

#endif // ROI_H