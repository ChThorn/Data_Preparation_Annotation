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
