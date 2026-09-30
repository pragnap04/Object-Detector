from ultralytics import YOLO
import cv2
import time


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 416
CAMERA_INDEX = 0


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()


# ============================================================
# FPS VARIABLES
# ============================================================

prev_time = time.time()


# ============================================================
# REAL-TIME DETECTION LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Run YOLO object detection and tracking
    results = model.track(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        persist=True,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # --------------------------------------------------------
    # OBJECT INFORMATION
    # --------------------------------------------------------

    object_count = len(results[0].boxes)

    detected_objects = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        if object_name not in detected_objects:
            detected_objects.append(object_name)

    object_text = ", ".join(detected_objects)

    if not object_text:
        object_text = "None"

    # --------------------------------------------------------
    # FPS CALCULATION
    # --------------------------------------------------------

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # --------------------------------------------------------
    # INFORMATION PANEL
    # --------------------------------------------------------

    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (520, 145),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Objects Detected: {object_count}",
        (25, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Detected: {object_text}",
        (25, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # DISPLAY FRAME
    # --------------------------------------------------------

    cv2.imshow(
        "Real-Time Object Detection",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Object detection stopped.")