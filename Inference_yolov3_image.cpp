#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>

class YoloDetector {
private:
    cv::dnn::Net net;
    float confThreshold;
    float nmsThreshold;
    
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);
    std::vector<std::string> getOutputsNames();

public:
    YoloDetector(const std::string& modelPath, 
                 const std::string& configPath,
                 float confidenceThreshold = 0.5,
                 float nmsThreshold = 0.4) {
        // Check if files exist
        std::ifstream modelFile(modelPath);
        if (!modelFile.good()) {
            throw std::runtime_error("Cannot open weights file: " + modelPath);
        }
        
        std::ifstream configFile(configPath);
        if (!configFile.good()) {
            throw std::runtime_error("Cannot open config file: " + configPath);
        }

        try {
            std::cout << "Loading YOLOv3 network...\n";
            std::cout << "Config: " << configPath << "\n";
            std::cout << "Weights: " << modelPath << "\n";
            
            net = cv::dnn::readNetFromDarknet(configPath, modelPath);
            if (net.empty()) {
                throw std::runtime_error("Failed to create network");
            }
            
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            
            this->confThreshold = confidenceThreshold;
            this->nmsThreshold = nmsThreshold;
            
            std::cout << "Network loaded successfully\n";
        }
        catch (const cv::Exception& e) {
            throw std::runtime_error("Failed to load the network: " + std::string(e.what()));
        }
    }

    cv::Mat detect(const cv::Mat& frame);
};

// Detect method implementation
cv::Mat YoloDetector::detect(const cv::Mat& frame) {
    cv::Mat blob;
    cv::Mat processedFrame = frame.clone();
    
    try {
        // Create a 4D blob from a frame
        cv::dnn::blobFromImage(frame, blob, 1/255.0, 
                              cv::Size(416, 416), 
                              cv::Scalar(0,0,0), 
                              true, false);
        
        // Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        std::vector<cv::Mat> outs;
        net.forward(outs, getOutputsNames());
        
        // Remove the bounding boxes with low confidence
        postprocess(processedFrame, outs);
        
        return processedFrame;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error during detection: " << e.what() << std::endl;
        return frame; // Return original frame in case of error
    }
}

// Postprocess method implementation
void YoloDetector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (const auto& out : outs) {
        float* data = (float*)out.data;
        for (int j = 0; j < out.rows; ++j, data += out.cols) {
            float confidence = data[4];  // Object confidence
            
            // For single class, we only need to check the first class probability
            float classProb = data[5];
            confidence *= classProb;
            
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                confidences.push_back(confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    // Draw Bounding boxes
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        
        // Draw a rectangle displaying the bounding box
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 3);
        
        // Display confidence score
        std::string label = cv::format("Confidence: %.2f", confidences[idx]);
        
        // Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                           0.5, 1, &baseLine);
        cv::rectangle(frame, 
                     cv::Point(box.x, box.y - labelSize.height),
                     cv::Point(box.x + labelSize.width, box.y + baseLine),
                     cv::Scalar(255, 255, 255), 
                     cv::FILLED);
        cv::putText(frame, label, 
                   cv::Point(box.x, box.y),
                   cv::FONT_HERSHEY_SIMPLEX, 
                   0.5, cv::Scalar(0, 0, 0));
    }
}

// GetOutputsNames method implementation
std::vector<std::string> YoloDetector::getOutputsNames() {
    static std::vector<std::string> names;
    if (names.empty()) {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}

// No Image path added at the running
// int main() {
//     try {
//         std::cout << "Starting YOLOv3 detection program...\n";
        
//         // Your absolute paths
//         std::string modelPath = "/home/thornch/Documents/YOLOv3_custom_data_and_onnx/yolov3_darknet_kimbap/darknet/backup/yolov3-kimbap_3000.weights";
//         std::string configPath = "/home/thornch/Documents/YOLOv3_custom_data_and_onnx/yolov3_darknet_kimbap/darknet/cfg/yolov3-kimbap.cfg";
//         std::string imagePath = "/home/thornch/Documents/Cpp/Kimbab_dataset/darknet_dataset_capture/images/train/11.jpg";
        
//         std::cout << "Checking files...\n";
        
//         // Initialize detector
//         YoloDetector detector(modelPath, configPath);
        
//         std::cout << "Loading image: " << imagePath << "\n";
//         // Read image
//         cv::Mat frame = cv::imread(imagePath);
//         if (frame.empty()) {
//             std::cerr << "Error: Could not read the image: " << imagePath << std::endl;
//             return -1;
//         }
        
//         std::cout << "Image loaded successfully. Size: " << frame.size() << "\n";
//         std::cout << "Performing detection...\n";
        
//         // Perform detection
//         cv::Mat result = detector.detect(frame);
        
//         std::cout << "Detection completed. Showing results...\n";
        
//         // Show result
//         cv::imshow("Object Detection", result);
//         cv::waitKey(0);
        
//         return 0;
//     }
//     catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
// }

// Need to add image path after the execute file compiled
int main(int argc, char** argv) {
    try {
        // Check if image path is provided as command line argument
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <image_path>\n";
            std::cerr << "Example: " << argv[0] << " /path/to/image.jpg\n";
            return -1;
        }

        std::string imagePath = argv[1];
        std::cout << "Starting YOLOv3 detection program...\n";
        
        // Model paths remain constant
        std::string modelPath = "/home/thornch/Documents/YOLOv3_custom_data_and_onnx/yolov3_darknet_kimbap/darknet/backup/yolov3-kimbap_3000.weights";
        std::string configPath = "/home/thornch/Documents/YOLOv3_custom_data_and_onnx/yolov3_darknet_kimbap/darknet/cfg/yolov3-kimbap.cfg";
        
        std::cout << "Checking files...\n";
        
        // Initialize detector
        YoloDetector detector(modelPath, configPath);
        
        std::cout << "Loading image: " << imagePath << "\n";
        // Read image
        cv::Mat frame = cv::imread(imagePath);
        if (frame.empty()) {
            std::cerr << "Error: Could not read the image: " << imagePath << std::endl;
            return -1;
        }
        
        std::cout << "Image loaded successfully. Size: " << frame.size() << "\n";
        std::cout << "Performing detection...\n";
        
        // Perform detection
        cv::Mat result = detector.detect(frame);
        
        std::cout << "Detection completed. Showing results...\n";
        
        // Show result
        cv::imshow("Object Detection", result);
        cv::waitKey(0);
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}