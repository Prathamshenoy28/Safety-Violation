
# Safety-Violation Detection System using YOLOv8

## Overview
This project utilizes a custom-trained YOLOv8 model for detecting personal protective equipment (PPE) violations in video frames. The system identifies violations such as missing hardhats, missing safety vests, and other safety gear in construction or industrial environments. It logs violations in a JSON file and displays results through a Streamlit dashboard.

The system processes a video, detects relevant objects, and logs violations based on the detection of specific objects in the frames. The video is annotated with bounding boxes and labels to visually highlight the detected objects. Additionally, screenshots of frames with violations are saved.

## Features
- **Object Detection**: Detects classes such as Hardhat, No-Hardhat, No-Safety Vest, Person, and Safety Vest.
- **Violation Detection**: Logs violations based on specific criteria (e.g., missing PPE).
- **Interval-Based Logging**: Violations are logged every few seconds to reduce noise and focus on significant events.
- **Bounding Boxes**: Draws bounding boxes around detected violations and labels the objects accordingly.
- **Screenshot Capture**: Captures screenshots of frames with violations.
- **JSON Logs**: Saves violation data (frame number, violation type, confidence score, bounding box coordinates, and screenshot path) to a JSON file.
- **Streamlit Dashboard**: Provides an interactive dashboard to visualize violations and inspect frames where PPE violations occurred.

## Requirements

- Python 3.x
- Dependencies:
  - `opencv-python`
  - `torch`
  - `ultralytics` (for YOLOv8)
  - `numpy`
  - `json`
  - `streamlit`
  
You can install the required packages using:

```bash
pip install opencv-python torch ultralytics numpy streamlit
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Prathamshenoy28/Safety-Violation.git
   ```

2. Place your trained YOLOv8 model (in `.onnx` format) in the appropriate directory (`C:/Users/HP/Downloads/your_output_directory2/runs/detect/yolov8s_ppe_css_50_epochs/weights/best.onnx`) or update the path in the script.

3. Place your input video in the correct directory (`C:/Users/HP/Downloads/your_output_directory2/videoeg/example_video.mp4`) or update the video path in the script.

## Running the Script

To run the video detection script, execute the following command in your terminal:

```bash
python main.py
```

This will process the video, detect PPE violations, and generate:
- An output video (`output_video_with_violations.mp4`) with bounding boxes and labels.
- A `violations.json` file containing details about each detected violation (frame number, violation type, confidence score, bounding box coordinates, and screenshot path).

## Streamlit Dashboard

1. Ensure that you have completed the setup and the `main.py` script has processed the video and generated the necessary output files (`output_video_with_violations.mp4` and `violations.json`).
  
2. To run the Streamlit app and visualize the violations, execute the following command:

   ```bash
   streamlit run app.py
   ```

3. This will open the Streamlit dashboard in your default web browser, where you can:
   - View the processed video with bounding boxes and labels.
   - Display detected violations and their details.
   - Inspect screenshots associated with each violation.

## Output

- **Output Video**: The annotated video showing bounding boxes and labels for all detected objects, including violations.
- **Violations Log**: A JSON file (`violations.json`) with detailed logs about the violations, including the following fields:
  - `frame_number`: Frame number in the video
  - `violation_type`: Type of violation (e.g., "NO-Hardhat")
  - `confidence`: Confidence score of the detection
  - `bbox`: Coordinates of the bounding box around the detected object
  - `screenshot`: Path to the saved screenshot for the violation

## Example

Here’s an example entry from the `violations.json` file:

```json
[
  {
    "frame_number": 150,
    "violation_type": "NO-Hardhat",
    "confidence": 0.92,
    "bbox": [50, 100, 150, 200],
    "screenshot": "./screenshots/frame_150_cls_2.jpg"
  },
  {
    "frame_number": 200,
    "violation_type": "NO-Safety Vest",
    "confidence": 0.87,
    "bbox": [100, 120, 180, 240],
    "screenshot": "./screenshots/frame_200_cls_4.jpg"
  }
]
```

## Customization

- **Confidence Threshold**: You can adjust the `CONFIDENCE_THRESHOLD` variable in the script to control the detection sensitivity.
- **Classes to Detect**: The `CLASSES_TO_DETECT` list allows you to specify which classes you want to detect in the video. Currently, the classes include Hardhat, NO-Hardhat, NO-Safety Vest, Person, and Safety Vest.
- **Detection Interval**: You can adjust the `interval_seconds` variable to control how often the system logs violations based on the frame rate.

## Troubleshooting

- **Error: Could not open video**: Make sure the video file path is correct, and the video file is accessible.
- **Low Detection Accuracy**: Ensure that the YOLOv8 model is properly trained and the confidence threshold is set appropriately.

