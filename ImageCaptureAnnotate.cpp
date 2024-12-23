#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <map>

namespace fs = std::filesystem;

class AutoAnnotator {
private:
    cv::dnn::Net net;
    std::vector<std::string> classes;
    std::map<std::string, int> class_map;
    float conf_threshold;
    float nms_threshold;
    
public:
    AutoAnnotator(const std::string& model_cfg,
                 const std::string& model_weights,
                 const std::string& class_file,
                 float conf_thresh = 0.5,
                 float nms_thresh = 0.4) 
        : conf_threshold(conf_thresh), nms_threshold(nms_thresh) {
        
        // Load pre-trained model (e.g., COCO trained model)
        net = cv::dnn::readNetFromDarknet(model_cfg, model_weights);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        
        // Load class names
        std::ifstream ifs(class_file);
        std::string line;
        int class_id = 0;
        while (std::getline(ifs, line)) {
            classes.push_back(line);
            class_map[line] = class_id++;
        }
    }
    
    struct Detection {
        cv::Rect box;
        float confidence;
        int class_id;
        std::string class_name;
    };

    std::vector<Detection> detectObjects(const cv::Mat& frame) {
        cv::Mat blob;
        std::vector<Detection> detections;
        
        // Create blob from image
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(416, 416), 
                              cv::Scalar(0,0,0), true, false);
        net.setInput(blob);
        
        // Get outputs
        std::vector<cv::Mat> outs;
        net.forward(outs, getOutputsNames());
        
        // Process detections
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        
        for (const auto& out : outs) {
            for (int i = 0; i < out.rows; ++i) {
                const float* data = (float*)out.row(i).data;
                cv::Mat scores = out.row(i).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                
                if (confidence > conf_threshold) {
                    Detection det;
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    
                    det.box = cv::Rect(centerX - width/2, centerY - height/2, 
                                     width, height);
                    det.confidence = confidence;
                    det.class_id = classIdPoint.x;
                    det.class_name = classes[det.class_id];
                    
                    detections.push_back(det);
                }
            }
        }
        
        return detections;
    }

private:
    std::vector<std::string> getOutputsNames() {
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
};

class AutomaticDatasetAnnotator {
private:
    rs2::pipeline pipe;
    rs2::config cfg;
    AutoAnnotator annotator;
    std::string dataset_path;
    std::string images_path;  // Added member variable
    std::string labels_path;  // Added member variable
    
    
public:
    AutomaticDatasetAnnotator(const std::string& base_path,
                            const std::string& model_cfg,
                            const std::string& model_weights,
                            const std::string& class_file)
        : annotator(model_cfg, model_weights, class_file) {
        
         // Get absolute path for dataset
        dataset_path = fs::absolute(base_path).string();
        images_path = dataset_path + "/images/train";
        labels_path = dataset_path + "/labels/train";
        
        // Create directories if they don't exist
        fs::create_directories(images_path);
        fs::create_directories(labels_path);
        
        // Print directory paths
        std::cout << "Dataset directory: " << dataset_path << std::endl;
        std::cout << "Images will be saved to: " << images_path << std::endl;
        std::cout << "Labels will be saved to: " << labels_path << std::endl;
        setupRealSense();
    }
    
    void setupRealSense() {
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        pipe.start(cfg);
    }
    
    void collectAndAnnotate(int num_frames) {
        cv::namedWindow("Auto Annotation", cv::WINDOW_AUTOSIZE);
        int frame_count = 0;
        
        while (frame_count < num_frames) {
            // Capture frame
            rs2::frameset frames = pipe.wait_for_frames();
            rs2::frame color_frame = frames.get_color_frame();
            cv::Mat frame(cv::Size(640, 480), CV_8UC3, 
                         (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            
            // Detect objects
            auto detections = annotator.detectObjects(frame);
            
            // Draw detections
            cv::Mat display = frame.clone();
            for (const auto& det : detections) {
                cv::rectangle(display, det.box, cv::Scalar(0, 255, 0), 2);
                std::string label = det.class_name + " " + 
                                  std::to_string(det.confidence).substr(0, 4);
                cv::putText(display, label, 
                           cv::Point(det.box.x, det.box.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(0, 255, 0), 2);
            }
            
            // Display info
            std::string info = "Frame: " + std::to_string(frame_count) + 
                             "/" + std::to_string(num_frames);
            cv::putText(display, info, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::putText(display, "SPACE: Save with annotations", 
                       cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 
                       0.5, cv::Scalar(0, 255, 0), 1);
            cv::putText(display, "R: Retry detection", 
                       cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 
                       0.5, cv::Scalar(0, 255, 0), 1);
            cv::putText(display, "Q: Quit", 
                       cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 
                       0.5, cv::Scalar(0, 255, 0), 1);
            
            cv::imshow("Auto Annotation", display);
            char key = cv::waitKey(1);
            
            if (key == ' ') {  // Space to save
                saveAnnotations(frame, detections, frame_count);
                frame_count++;
            }
            else if (key == 'r') {  // Retry detection
                continue;
            }
            else if (key == 'q') {  // Quit
                break;
            }
        }
        
        cv::destroyAllWindows();
    }
    
private:
    void saveAnnotations(const cv::Mat& frame, 
                        const std::vector<AutoAnnotator::Detection>& detections,
                        int frame_count) {
        // Generate filenames with absolute paths
        std::string img_filename = images_path + "/" + 
                                 std::to_string(frame_count) + ".jpg";
        std::string label_filename = labels_path + "/" + 
                                   std::to_string(frame_count) + ".txt";
        
        // Save image
        cv::imwrite(img_filename, frame);
        
        // Save annotations in YOLO format
        std::ofstream label_file(label_filename);
        
        for (const auto& det : detections) {
            float x_center = (det.box.x + det.box.width/2.0f) / frame.cols;
            float y_center = (det.box.y + det.box.height/2.0f) / frame.rows;
            float width = (float)det.box.width / frame.cols;
            float height = (float)det.box.height / frame.rows;
            
            label_file << det.class_id << " " 
                      << x_center << " " 
                      << y_center << " "
                      << width << " " 
                      << height << "\n";
        }
        
        label_file.close();
        
        // Print saved file locations
        std::cout << "Saved image to: " << img_filename << std::endl;
        std::cout << "Saved labels to: " << label_filename << std::endl;
    }
};

int main() 
{
    try {
        // Get current working directory
        std::string current_path = fs::current_path().string();
        std::cout << "Current working directory: " << current_path << std::endl;

        AutomaticDatasetAnnotator annotator(
            "darknet_dataset",
            "yolov3.cfg",
            "yolov3.weights",
            "coco.names"
        );

        std::cout << "\nPress:\n";
        std::cout << "SPACE - Save frame with annotations\n";
        std::cout << "R     - Retry detection\n";
        std::cout << "Q     - Quit\n\n";
        
        annotator.collectAndAnnotate(100);  // Collect 100 frames
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}