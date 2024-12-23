#include "roi.h"
#include <stdexcept>

ROIBox::ROIBox() : roi(0, 0, 0, 0) {}

ROIBox::ROIBox(int x, int y, int width, int height) : roi(x, y, width, height) {
    validateROI();
}

ROIBox::ROIBox(const cv::Rect& rect) : roi(rect) {
    validateROI();
}

void ROIBox::setROI(int x, int y, int width, int height) {
    roi = cv::Rect(x, y, width, height);
    validateROI();
}

void ROIBox::setROI(const cv::Rect& rect) {
    roi = rect;
    validateROI();
}

void ROIBox::validateROI() {
    if (roi.width <= 0 || roi.height <= 0) {
        throw std::invalid_argument("ROI dimensions must be positive");
    }
}

bool ROIBox::isWithinFrame(const cv::Mat& frame) const {
    return roi.x >= 0 && roi.y >= 0 && 
           roi.x + roi.width <= frame.cols && 
           roi.y + roi.height <= frame.rows;
}

cv::Mat ROIBox::extractROI(const cv::Mat& frame) const {
    if (!isWithinFrame(frame)) {
        throw std::runtime_error("ROI is outside frame boundaries");
    }
    return frame(roi).clone();
}

cv::Rect ROIBox::clipRectToROI(const cv::Rect& rect) const {
    // Convert rect to ROI coordinates
    cv::Rect localRect = rect;
    localRect.x -= roi.x;
    localRect.y -= roi.y;
    
    // Clip to ROI boundaries
    cv::Rect clipped;
    clipped.x = std::max(localRect.x, 0);
    clipped.y = std::max(localRect.y, 0);
    clipped.width = std::min(localRect.width, roi.width - clipped.x);
    clipped.height = std::min(localRect.height, roi.height - clipped.y);
    
    return clipped;
}

void ROIBox::draw(cv::Mat& frame, const cv::Scalar& color, int thickness) const {
    if (!isWithinFrame(frame)) {
        throw std::runtime_error("Cannot draw ROI: outside frame boundaries");
    }
    cv::rectangle(frame, roi, color, thickness);
}