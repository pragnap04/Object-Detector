# Real-Time Object Detection System

A real-time computer vision application built with **Python, OpenCV, and YOLO11** that detects and tracks multiple objects through a webcam.

The system displays bounding boxes, object labels, confidence information, object count, tracking results, and real-time FPS.

---

## 🚀 Features

- Real-time object detection
- Multi-object tracking
- Bounding boxes and object labels
- Confidence-based detection
- Object counting
- Detected object information
- Real-time FPS monitoring
- CPU-optimized inference
- Live webcam processing
- Simple keyboard control

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| YOLO11 | Object detection and tracking |
| Ultralytics | YOLO framework |
| OpenCV | Webcam and image processing |
| PyTorch | Deep learning backend |

---

## 🧠 How It Works

```text
              Webcam
                 │
                 ▼
          Capture Video Frame
                 │
                 ▼
          YOLO11 Detection
                 │
                 ▼
           Object Tracking
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
    Bounding   Object    Object
     Boxes     Count      Names
        │        │        │
        └────────┼────────┘
                 ▼
          FPS Calculation
                 │
                 ▼
        Annotated Live Frame