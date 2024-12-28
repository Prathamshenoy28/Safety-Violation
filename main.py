import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import json


class CFG:
    WEIGHTS = 'C:/Users/HP/Downloads/your_output_directory2/runs/detect/yolov8s_ppe_css_50_epochs/weights/best.onnx'
    CONFIDENCE = 0.60
    CLASSES_TO_DETECT = [0, 2, 4, 5, 7]  # Hardhat, NO-Hardhat, NO-Safety Vest, Person, Safety Vest
    VID_001 = 'C:/Users/HP/Downloads/your_output_directory2/videoeg/example_video.mp4'
    PATH_TO_INFER_ON = VID_001
    EXT = PATH_TO_INFER_ON.split('.')[-1]
    FILENAME_TO_INFER_ON = PATH_TO_INFER_ON.split('/')[-1].split('.')[0]
    ROOT_DIR = 'C:/Users/HP/Downloads/your_output_directory2/your_output_directory/videoeg'
    OUTPUT_DIR = './'


class_names = ["Hardhat", "NO-Hardhat", "NO-Safety Vest", "Person", "Safety Vest"]
relevant_indices = {
    0: "Hardhat",
    2: "NO-Hardhat",
    4: "NO-Safety Vest",
    5: "Person",
    7: "Safety Vest"
}


CONFIDENCE_THRESHOLD = 0.7  
violation_events = []  
tracked_people = {}  

tracked_violations = {}
COOLDOWN_TIME_SECONDS = 0.5  

def process_frame(frame, frame_number, fps):
    global tracked_violations
    results = model.predict(source=frame, device='cpu')
    detections = results[0].boxes.data.cpu().numpy()
    
    
    cooldown_frames = int(fps * COOLDOWN_TIME_SECONDS)
    
    for detection in detections:
        cls = int(detection[5])  
        conf = detection[4]  
        bbox = detection[:4]  

        
        if cls in relevant_indices and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 0, 255) if cls in [2, 4] else (0, 255, 0) 
            label = f"{relevant_indices[cls]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        
        if cls in [2, 4] and conf >= CONFIDENCE_THRESHOLD:
            violation_type = relevant_indices[cls]
            person_id = hash(tuple(bbox))  

            
            if person_id not in tracked_violations or frame_number - tracked_violations[person_id] > cooldown_frames:
                # Log the violation
                violation_events.append({
                    "frame_number": int(frame_number),
                    "violation_type": violation_type,
                    "confidence": float(round(conf, 2)),
                    "bbox": [int(coord) for coord in bbox.tolist()]
                })

               
                tracked_violations[person_id] = frame_number

    return frame


model = YOLO(CFG.WEIGHTS)
input_video = CFG.PATH_TO_INFER_ON
output_video = './output_video_with_violations.mp4'


cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))


frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    
    frame_with_boxes = process_frame(frame, frame_number, fps)
    out.write(frame_with_boxes)

import json

with open('violations.json', 'w') as f:
    json.dump(violation_events, f, indent=4)
print("Violations saved to violations.json")

cap.release()
out.release()
print(f'Processed video saved to: {output_video}')


