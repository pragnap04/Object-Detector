import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("road_sign_classifier.keras")

# Class names used during training
CLASS_NAMES = ["trafficlight", "stop", "other"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access the camera.")
        break

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to the model's input size
    img = cv2.resize(rgb, (128, 128))

    # Normalize pixel values
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    class_id = np.argmax(prediction[0])
    confidence = prediction[0][class_id]

    label = CLASS_NAMES[class_id]

    # Display prediction
    text = f"{label} ({confidence * 100:.1f}%)"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Road Sign Classifier", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()