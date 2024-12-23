#include "roi.h"
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

// Function to read class names from coco.names
std::vector<std::string> loadClassNames(const std::string& filename) {
    std::vector<std::string> classNames;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

// Function to draw the detected objects
void drawDetections(cv::Mat& frame, const ROIBox& roiBox,
                   const std::vector<cv::Rect>& boxes,
                   const std::vector<int>& classIds,
                   const std::vector<float>& confidences,
                   const std::vector<std::string>& classNames) {
    // Draw ROI
    roiBox.draw(frame);

    cv::Rect roi = roiBox.getROI();
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        // Clip detection to ROI and convert to frame coordinates
        cv::Rect clippedBox = roiBox.clipRectToROI(boxes[i]);
        clippedBox.x += roi.x;
        clippedBox.y += roi.y;

        // Generate random color for this class
        cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
        
        // Draw detection box
        cv::rectangle(frame, clippedBox, color, 2);
        
        // Prepare label
        std::string label = classNames[classIds[i]] + ": " + 
                           cv::format("%.2f", confidences[i]);
        
        // Calculate label position
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                           0.5, 1, &baseLine);
        
        cv::Point labelPos(clippedBox.x, clippedBox.y);
        
        // Ensure label stays within ROI
        if (labelPos.y - labelSize.height - baseLine - 10 < roi.y) {
            labelPos.y = clippedBox.y + clippedBox.height + labelSize.height;
        } else {
            labelPos.y = clippedBox.y - baseLine - 5;
        }
        
        // Draw label background and text
        cv::rectangle(frame, 
                     cv::Point(labelPos.x, labelPos.y - labelSize.height - baseLine - 5),
                     cv::Point(labelPos.x + labelSize.width, labelPos.y),
                     color, cv::FILLED);
        cv::putText(frame, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, cv::Scalar(255, 255, 255), 1);
    }
}

int main() {
    try {
        // Initialize YOLO model
        std::string modelConfig = "yolov3.cfg";
        std::string modelWeights = "yolov3.weights";
        std::string classFile = "coco.names";

        // Load class names
        std::vector<std::string> classNames = loadClassNames(classFile);

        // Load network
        cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // Read input image
        cv::Mat frame = cv::imread("zidane.jpg");
        if (frame.empty()) {
            throw std::runtime_error("Failed to load image!");
        }

        // Create and set ROI
        ROIBox roiBox;
        roiBox.setROI(180, 100, 500, 610);
        cv::Rect roi = roiBox.getROI();

        // Create a black image same size as original
        cv::Mat blackImage = cv::Mat::zeros(frame.size(), frame.type());
        
        // Copy ONLY the ROI content to the black image
        frame(roi).copyTo(blackImage(roi));

        // Now run detection on the modified image
        cv::Mat blob;
        cv::dnn::blobFromImage(blackImage, blob, 1/255.0, cv::Size(416, 416), 
                              cv::Scalar(0,0,0), true, false, CV_32F);

        // Run network on modified image
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Process detections
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;
        float confThreshold = 0.2;

        for (const auto& output : outputs) {
            for (int i = 0; i < output.rows; ++i) {
                const float* data = output.ptr<float>(i);
                float confidence = data[4];

                if (confidence > confThreshold) {
                    cv::Mat scores = output.row(i).colRange(5, output.cols);
                    cv::Point classIdPoint;
                    double maxScore;
                    cv::minMaxLoc(scores, nullptr, &maxScore, nullptr, &classIdPoint);

                    if (maxScore > confThreshold) {
                        // Get detection coordinates
                        int centerX = static_cast<int>(data[0] * blackImage.cols);
                        int centerY = static_cast<int>(data[1] * blackImage.rows);
                        int width = static_cast<int>(data[2] * blackImage.cols);
                        int height = static_cast<int>(data[3] * blackImage.rows);
                        int left = centerX - width/2;
                        int top = centerY - height/2;

                        // Only accept detection if it's within ROI
                        cv::Point center(centerX, centerY);
                        if (roi.contains(center)) {
                            boxes.push_back(cv::Rect(left, top, width, height));
                            confidences.push_back(static_cast<float>(maxScore));
                            classIds.push_back(classIdPoint.x);
                        }
                    }
                }
            }
        }

        // Apply NMS
        std::vector<int> indices;
        float nmsThreshold = 0.4;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        }

        // Draw ROI box
        cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2);

        // Draw valid detections with clipping
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];

            // Only draw if box center is in ROI
            cv::Point boxCenter(box.x + box.width/2, box.y + box.height/2);
            if (roi.contains(boxCenter)) {
                // Clip the detection box to ROI boundaries
                cv::Rect clippedBox = box & roi;  // Intersection with ROI
                
                cv::Scalar color(0, 255, 0);
                cv::rectangle(frame, clippedBox, color, 2);

                // Adjust label position to stay within ROI
                std::string label = classNames[classIds[idx]] + ": " + 
                                  cv::format("%.2f", confidences[idx]);

                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                   0.5, 1, &baseLine);

                // Calculate label position to ensure it stays within ROI
                cv::Point labelPos;
                labelPos.x = std::max(clippedBox.x, roi.x);
                labelPos.y = std::max(clippedBox.y - baseLine - 5, roi.y + labelSize.height);

                // Ensure label width doesn't exceed ROI
                if (labelPos.x + labelSize.width > roi.x + roi.width) {
                    labelPos.x = roi.x + roi.width - labelSize.width;
                }

                // Ensure label background stays within ROI
                cv::Rect labelBackground(
                    cv::Point(labelPos.x, labelPos.y - labelSize.height - baseLine - 5),
                    cv::Point(labelPos.x + labelSize.width, labelPos.y)
                );
                labelBackground = labelBackground & roi;  // Clip to ROI

                cv::rectangle(frame, labelBackground, color, cv::FILLED);
                cv::putText(frame, label, 
                           cv::Point(labelBackground.x, labelBackground.y + labelBackground.height - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        }

        // Show debug images
        cv::imshow("Black Image with ROI", blackImage);
        cv::imshow("Final Result", frame);
        cv::waitKey(0);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

// int main() {
//     try {
//         // Initialize YOLO model
//         std::string modelConfig = "yolov3.cfg";
//         std::string modelWeights = "yolov3.weights";
//         std::string classFile = "coco.names";

//         // Load class names
//         std::vector<std::string> classNames = loadClassNames(classFile);

//         // Load network
//         cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
//         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

//         // Read input image
//         cv::Mat frame = cv::imread("zidane.jpg");
//         if (frame.empty()) {
//             throw std::runtime_error("Failed to load image!");
//         }

//         // Create and set ROI
//         ROIBox roiBox;
//         // Example: Set ROI using pixel coordinates
//         roiBox.setROI(140, 100, 250, 200);  // x, y, width, height

//         if (!roiBox.isWithinFrame(frame)) {
//             throw std::runtime_error("ROI is outside frame boundaries");
//         }

//         // Extract ROI for detection
//         cv::Mat roiMat = roiBox.extractROI(frame);

//         // Create blob from ROI
//         cv::Mat blob;
//         cv::dnn::blobFromImage(roiMat, blob, 1/255.0, cv::Size(416, 416), 
//                               cv::Scalar(0,0,0), true, false, CV_32F);

//         // Detect objects
//         net.setInput(blob);
//         std::vector<cv::Mat> outputs;
//         net.forward(outputs, net.getUnconnectedOutLayersNames());

//         // Initialize vectors for detection results
//         std::vector<int> classIds;
//         std::vector<float> confidences;
//         std::vector<cv::Rect> boxes;
//         float confThreshold = 0.5;

//         // Process detections
//         for (const auto& output : outputs) {
//             for (int i = 0; i < output.rows; ++i) {
//                 const float* data = output.ptr<float>(i);
//                 float confidence = data[4];

//                 if (confidence > confThreshold) {
//                     cv::Mat scores = output.row(i).colRange(5, output.cols);
//                     cv::Point classIdPoint;
//                     double maxScore;
//                     cv::minMaxLoc(scores, nullptr, &maxScore, nullptr, &classIdPoint);

//                     if (maxScore > confThreshold) {
//                         int centerX = static_cast<int>(data[0] * roiMat.cols);
//                         int centerY = static_cast<int>(data[1] * roiMat.rows);
//                         int width = static_cast<int>(data[2] * roiMat.cols);
//                         int height = static_cast<int>(data[3] * roiMat.rows);
//                         int left = centerX - width/2;
//                         int top = centerY - height/2;

//                         classIds.push_back(classIdPoint.x);
//                         confidences.push_back(static_cast<float>(maxScore));
//                         boxes.push_back(cv::Rect(left, top, width, height));
//                     }
//                 }
//             }
//         }

//         // Perform Non-Maximum Suppression
//         std::vector<int> indices;
//         float nmsThreshold = 0.4;
//         cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

//         // Prepare final detections
//         std::vector<cv::Rect> nmsBoxes;
//         std::vector<int> nmsClassIds;
//         std::vector<float> nmsConfidences;

//         for (int idx : indices) {
//             nmsBoxes.push_back(boxes[idx]);
//             nmsClassIds.push_back(classIds[idx]);
//             nmsConfidences.push_back(confidences[idx]);
//         }

//         // Draw detections
//         drawDetections(frame, roiBox, nmsBoxes, nmsClassIds, nmsConfidences, classNames);

//         // Display results
//         std::cout << "Found " << nmsBoxes.size() << " objects in ROI" << std::endl;
//         cv::imshow("Object Detection with ROI", frame);
//         cv::waitKey(0);

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }

//     return 0;
// }