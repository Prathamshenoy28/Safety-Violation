import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import json

class CFG:
    WEIGHTS = 'C:/Users/HP/Downloads/your_output_directory2/runs/detect/yolov8s_ppe_css_50_epochs/weights/best.onnx'
    CONFIDENCE = 0.8
    CLASSES_TO_DETECT = [0, 2, 4, 5, 7]  
    VID_001 = 'C:/Users/HP/Downloads/your_output_directory2/videoeg/example_video.mp4'
    PATH_TO_INFER_ON = VID_001
    EXT = PATH_TO_INFER_ON.split('.')[-1]
    FILENAME_TO_INFER_ON = PATH_TO_INFER_ON.split('/')[-1].split('.')[0]
    ROOT_DIR = 'C:/Users/HP/Downloads/your_output_directory2/your_output_directory/videoeg'
    OUTPUT_DIR = './screenshots/'  

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

relevant_indices = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    4: "NO-Safety Vest",
    5: "Person",
    7: "Safety Vest"
}

CONFIDENCE_THRESHOLD = 0.85  
previous_person_count = -1  
violation_events = []  


model = YOLO(CFG.WEIGHTS)


fps = 30  
interval_seconds = 5 
frames_per_interval = int(fps * interval_seconds)  

def process_frame(frame, frame_number):
    global previous_person_count
    results = model.predict(source=frame, device='cpu')
    detections = results[0].boxes.data.cpu().numpy()

    frame_violations = []  

   
    for detection in detections:
        cls = int(detection[5])  
        conf = detection[4]  
        bbox = detection[:4]  

        
        if cls not in relevant_indices:
            continue

        if conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, bbox)
            
            color = (0, 255, 0) if cls in [0, 7, 5] else (0, 0, 255)
            label = f"{relevant_indices[cls]}: {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cls in [2, 4]:  
                screenshot_path = os.path.join(CFG.OUTPUT_DIR, f"frame_{frame_number}_cls_{cls}.jpg")
                cv2.imwrite(screenshot_path, frame)

                frame_violations.append({
                    "frame_number": int(frame_number),
                    "violation_type": relevant_indices[cls],
                    "confidence": float(round(conf, 2)),
                    "bbox": [int(coord) for coord in bbox.tolist()],
                    "screenshot": screenshot_path
                })

   
    if frame_number % frames_per_interval == 0:
        current_person_count = sum(1 for detection in detections if int(detection[5]) == 5)

        if current_person_count != previous_person_count:
            previous_person_count = current_person_count
            violation_events.extend(frame_violations)  

    return frame

input_video = CFG.PATH_TO_INFER_ON
output_video = './output_video_with_violations.mp4'
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
   
    frame_with_boxes = process_frame(frame, frame_number)

    out.write(frame_with_boxes)

with open('violations.json', 'w') as f:
    json.dump(violation_events, f, indent=4)


cap.release()
out.release()
print(f'Processed video saved to: {output_video}')
