#!/bin/bash

set -e

echo "Cloning YOLOv5 repository (v6.0)..."
if [ ! -d "yolov5" ]; then
    git clone -b v6.0 https://github.com/ultralytics/yolov5.git
fi
cd yolov5

echo "Installing required Python packages..."
pip install -r requirements.txt

echo "Downloading sample image..."
if [ ! -f "../zidane.jpg" ]; then
    wget -O ../zidane.jpg https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg
fi

cd ..

echo "Creating Python inference script..."
cat > yolo_inference.py << EOL
import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load image
img = cv2.imread('zidane.jpg')

# Inference
results = model(img)

# Results
results.print()  # Print results to screen
results.save()  # Save results to 'runs/detect/exp'

# Get bounding boxes
boxes = results.xyxy[0].cpu().numpy()

# Draw bounding boxes on the image
for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    if conf > 0.5:  # Confidence threshold
        color = (0, 255, 0)  # Green color for the rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{results.names[int(cls)]}: {conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the image with bounding boxes
cv2.imwrite('result.jpg', img)

print("Inference complete. Results saved to 'result.jpg'.")
EOL

echo "Creating C++ display script..."
cat > display_result.cpp << EOL
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
EOL

echo "Running Python inference..."
python yolo_inference.py

echo "Compiling C++ display script..."
g++ -std=c++11 display_result.cpp -o display_result `pkg-config --cflags --libs opencv4`

echo "Running C++ display script..."
./display_result