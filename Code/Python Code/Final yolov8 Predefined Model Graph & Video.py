import torch
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from ultralytics import YOLO
from picamera2 import Picamera2
from time import sleep
import matplotlib.pyplot as plt
from collections import Counter

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')

# Load emotion classification model
with open('saved models/model_EfficientNetB3.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("saved models/model_EfficientNetB3_.weights.h5")

# Emotion labels
emotion_labels = [
    'safe driving', 'texting right', 'talking on the phone - right', 'texting left',
    'talking on the phone - left', 'operating the radio', 'drinking', 
    'reaching behind', 'hair and makeup', 'talking to passenger'
]

# Initialize emotion counter to track detected emotions
emotion_counter = Counter()

# Function to save the final emotion detection graph
def save_final_graph(emotion_counter):
    """Save the final graph to a file when exiting the application."""
    if not emotion_counter:  # If no emotions were detected
        print("No emotions detected, no graph to save.")
        return

    fig, ax = plt.subplots()
    emotions = list(emotion_counter.keys())
    counts = list(emotion_counter.values())
    
    bars = ax.bar(emotions, counts)
    
    # Add annotations to each bar (show the count above the bar)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')  # Add count label
    
    plt.xticks(rotation=45)
    plt.ylabel("Counts")
    plt.title("Final Emotion Detection Count")
    
    # Save the figure to a file
    plt.tight_layout()  # Ensure the labels don't get cut off
    plt.savefig("emotion_detection_counts.png")
    plt.close()  # Close the plot after saving to free resources
    print("Graph exported as 'emotion_detection_counts.png'.")

# Initialize the Picamera2 object
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1640, 1232)})
picam2.configure(config)

# Initialize the video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
out = cv2.VideoWriter('emotion_detection_output_yolo.avi', fourcc, 30.0, (1640, 1232))  # Output file

# Start the camera
picam2.start()

# Give the camera time to adjust before capturing frames
sleep(2)

try:
    # Loop to capture frames and perform YOLOv8 detection and emotion classification
    while True:
        # Capture frame using Picamera2
        frame = picam2.capture_array()

        # Use YOLOv8 to detect objects (e.g., faces)
        results = yolo_model(frame)

        # Extract the bounding boxes and class predictions from YOLOv8 results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2) format
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])  # Extract the box coordinates
                confidence = confidences[i]

                # Only process detections that have a high confidence (optional threshold)
                if confidence > 0.5:  # Adjust the threshold as needed
                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Extract and preprocess the face region for emotion classification
                    roi_rgb = frame[y1:y2, x1:x2]  # Crop the face region
                    if roi_rgb.size == 0:  # Skip invalid regions
                        continue
                    roi_rgb = cv2.resize(roi_rgb, (320, 240))  # Resize for emotion classification model
                    roi_rgb = roi_rgb.astype("float") / 255.0  # Normalize the pixel values
                    roi_rgb = img_to_array(roi_rgb)
                    roi_rgb = np.expand_dims(roi_rgb, axis=0)

                    # Predict emotion using EfficientNetB3
                    preds = model.predict(roi_rgb)[0]
                    label = emotion_labels[np.argmax(preds)]

                    # Update emotion counter
                    emotion_counter[label] += 1

                    # Display the label and confidence score
                    label_text = f"{label} {confidence:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame with YOLOv8 detections and emotion labels
        cv2.imshow('Driver Behavior Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Ensure proper cleanup and resource release
    print("Stopping camera and releasing resources.")
    picam2.stop()
    cv2.destroyAllWindows()

    # Release the video writer
    out.release()

    # Save the final emotion graph when exiting
    save_final_graph(emotion_counter)
