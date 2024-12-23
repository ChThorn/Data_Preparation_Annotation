#!/bin/bash

set -e

echo "Cloning YOLOv5 repository..."
if [ ! -d "yolov5" ]; then
    git clone -b v5.0 https://github.com/ultralytics/yolov5.git
fi
cd yolov5

echo "Installing required Python packages..."
pip install -r requirements.txt

echo "Exporting YOLOv5s model to ONNX format..."
python export.py --weights yolov5s.pt --include onnx

echo "Moving ONNX file to working directory..."
mv yolov5s.onnx ..
cd ..

echo "Downloading COCO class names..."
if [ ! -f "coco.names" ]; then
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
fi

echo "Downloading sample image..."
if [ ! -f "zidane.jpg" ]; then
    wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg
fi

echo "Creating C++ inference and display script..."
cat > yolov5_detection.cpp << EOL
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>

std::vector<std::string> load_class_list(const std::string& filename) {
    std::vector<std::string> class_list;
    std::ifstream ifs(filename);
    std::string line;
    while (std::getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void detect_objects(cv::Mat& image, cv::dnn::Net& net, const std::vector<std::string>& class_list) {
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = float(image.cols) / 640.0;
    float y_factor = float(image.rows) / 640.0;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outputs[0].rows; ++i) {
        float* row = (float*)outputs[0].row(i).data;
        float confidence = row[4];
        if (confidence >= 0.5) {
            float* classes_scores = row + 5;
            cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > 0.5) {
                float x = row[0];
                float y = row[1];
                float w = row[2];
                float h = row[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                class_ids.push_back(class_id.x);
                confidences.push_back(confidence);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];
        float conf = confidences[idx];

        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

        std::string label = cv::format("%s: %.2f", class_list[class_id].c_str(), conf);
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::putText(image, label, cv::Point(box.x, box.y - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }
}

int main() {
    std::vector<std::string> class_list = load_class_list("coco.names");
    cv::dnn::Net net = cv::dnn::readNetFromONNX("yolov5s.onnx");

    cv::Mat frame = cv::imread("zidane.jpg");
    if (frame.empty()) {
        std::cerr << "Error: Unable to read the image file." << std::endl;
        return -1;
    }

    detect_objects(frame, net, class_list);

    cv::imshow("YOLOv5 Detection", frame);
    cv::waitKey(0);

    return 0;
}
EOL

echo "Compiling C++ script..."
g++ -std=c++11 -O3 yolov5_detection.cpp -o yolov5_detectioncpp `pkg-config --cflags --libs opencv4`

echo "Running YOLOv5 detection..."
./yolov5_detectioncpp