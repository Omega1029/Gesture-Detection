import cv2
import numpy as np
import pickle

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model

# Load the labels from label.pickle
labels = ["call", "fist", "like", "ok", "palm", "peace_inverted", "stop", "three", "two_up",
          "dislike", "four", "mute", "one", "peace", "rock", "stop_inverted", "three2", "two_up_inverted"]

lb = LabelBinarizer()
binarized_labels = lb.fit_transform(labels)


# Load the trained model
model = load_model("model.h5")


def preprocess_frame(frame):
    # Preprocess the frame for your model (resize, normalize, etc.)
    frame = cv2.resize(frame, (96, 96))  # Adjust size based on your model
    #frame = frame / 255.0  # Normalize if required
    return np.expand_dims(frame, axis=0)  # Add batch dimension


def detect_objects():
    cap = cv2.VideoCapture(0)  # Use the default camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_frame = preprocess_frame(frame)

        # Make predictions
        predictions = model.predict(input_frame)
        print(predictions)
        confidence_threshold = 0.5
        for i in range(len(predictions)):
            # Get the confidence scores for each class
            confidence_scores = predictions[i]

            # Get the class ID with the highest confidence
            class_id = np.argmax(confidence_scores)
            confidence = confidence_scores[class_id]

            if confidence > confidence_threshold:
                # Draw a bounding box (dummy values here for illustration)
                # Replace these with your actual bounding box logic
                x, y, width, height = 100, 100, 50, 50  # Dummy values

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # Get the label for the current class
                label = labels[class_id]
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the object detection on live camera
detect_objects()
