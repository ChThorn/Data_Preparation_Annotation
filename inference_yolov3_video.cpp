#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
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
                 float confidenceThreshold = 0.8,
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

    cv::Mat detect(const cv::Mat& frame) {
        cv::Mat blob;
        cv::Mat processedFrame = frame.clone();
        
        try {
            cv::dnn::blobFromImage(frame, blob, 1/255.0, 
                                  cv::Size(416, 416), 
                                  cv::Scalar(0,0,0), 
                                  true, false);
            
            net.setInput(blob);
            
            std::vector<cv::Mat> outs;
            net.forward(outs, getOutputsNames());
            
            postprocess(processedFrame, outs);
            
            return processedFrame;
        }
        catch (const cv::Exception& e) {
            std::cerr << "Error during detection: " << e.what() << std::endl;
            return frame;
        }
    }
};

// Postprocess method implementation
void YoloDetector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (const auto& out : outs) {
        float* data = (float*)out.data;
        for (int j = 0; j < out.rows; ++j, data += out.cols) {
            float confidence = data[4];
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
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 3);
        
        std::string label = cv::format("Confidence: %.2f", confidences[idx]);
        
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

int main(int argc, char** argv) {
    try {
        std::cout << "Starting YOLOv3 detection program with RealSense...\n";
        
        // Model paths (update these to your actual paths)
        std::string modelPath = "/home/thornch/Documents/YOLOv3_custom_data_and_onnx/yolov3_darknet_kimbap/darknet/backup/yolov3-kimbap_3000.weights"; //
        std::string configPath = "/home/thornch/Documents/YOLOv3_custom_data_and_onnx/yolov3_darknet_kimbap/darknet/cfg/yolov3-kimbap.cfg";
        
        // Initialize RealSense pipeline
        rs2::pipeline pipe;
        rs2::config cfg;
        
        // Configure stream
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        
        // Start pipeline with configuration
        pipe.start(cfg);
        
        // Initialize detector
        YoloDetector detector(modelPath, configPath);
        
        std::cout << "RealSense and YOLO initialized successfully. Starting detection...\n";
        
        while (true) {
            // Wait for frames
            rs2::frameset frames = pipe.wait_for_frames();
            
            // Get color frame
            rs2::frame color_frame = frames.get_color_frame();
            
            // Convert RealSense frame to OpenCV Mat
            cv::Mat frame(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            
            if (frame.empty()) {
                std::cerr << "Error: Failed to capture frame from RealSense" << std::endl;
                continue;
            }
            
            // Perform detection
            cv::Mat result = detector.detect(frame);
            
            // Show result
            cv::imshow("RealSense Object Detection", result);
            
            // Break loop with 'q'
            char key = (char)cv::waitKey(1);
            if (key == 'q' || key == 27) {
                break;
            }
        }
        
        // Stop the pipeline
        pipe.stop();
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}