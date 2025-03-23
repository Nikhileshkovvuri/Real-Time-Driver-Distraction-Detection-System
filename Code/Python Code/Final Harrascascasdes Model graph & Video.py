import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from picamera2 import Picamera2
from time import sleep
import matplotlib.pyplot as plt
from collections import Counter

# Load emotion classification model
with open('saved models/model_EfficientNetB3.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("saved models/model_EfficientNetB3_.weights.h5")

# Load the face detector
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = [
    'safe driving', 'texting right', 'talking on the phone - right', 'texting left',
    'talking on the phone - left', 'operating the radio', 'drinking', 
    'reaching behind', 'hair and makeup', 'talking to passenger'
]

# Initialize the emotion counter
emotion_counter = Counter()

# Function to save the final emotion detection graph
def save_final_graph(emotion_counter):
    """Save the final graph to a file when exiting the application."""
    fig, ax = plt.subplots()
    emotions = list(emotion_counter.keys())
    counts = list(emotion_counter.values())
    
    # Ensure there is data to plot
    if len(emotions) == 0:
        print("No emotions detected, skipping graph generation.")
        return
    
    # Plot the bar chart
    ax.bar(emotions, counts)
    plt.xticks(rotation=45)
    plt.ylabel("Counts")
    plt.title("Final Emotion Detection Count")
    plt.tight_layout()  # Adjust layout to fit all labels
    # Save the figure to a file
    plt.savefig("emotion_detection_counts.png")
    plt.close()  # Close the plot after saving to free resources
    print("Graph exported as 'emotion_detection_counts.png'.")

# Initialize the Picamera2 object
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1640, 1232)})
picam2.configure(config)

# Initialize the video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
out = cv2.VideoWriter('emotion_detection_output_1.avi', fourcc, 20.0, (1640, 1232))  # Output file

# Start the camera
picam2.start()

# Give the camera time to adjust before capturing frames
sleep(2)

try:
    # Loop to capture frames and detect behavior
    while True:
        frame = picam2.capture_array()

        # Convert the captured frame to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract the Region of Interest (ROI) for emotion classification
            roi_gray = gray_image[y:y+h, x:x+w]
            roi_rgb = cv2.resize(roi_gray, (320, 240))  # Resize to 320x240 to match the model's input
            roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_GRAY2RGB)  # Convert to RGB

            # Preprocess the ROI for the model
            roi_rgb = roi_rgb.astype("float") / 255.0
            roi_rgb = img_to_array(roi_rgb)
            roi_rgb = np.expand_dims(roi_rgb, axis=0)

            # Make predictions on the ROI
            preds = model.predict(roi_rgb)[0]
            print("Predictions:", preds)  # Debugging: Check predictions

            # Get the label and confidence score
            label = emotion_labels[np.argmax(preds)]
            confidence = np.max(preds)
            print(f"Detected: {label}, Confidence: {confidence}")  # Debugging: Check label and confidence

            # Update the emotion counter
            emotion_counter[label] += 1

            # Display the prediction and confidence score on the frame
            if confidence > 0.5:  # Only display if confidence is above 0.5
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Write the frame to the video file
        out.write(frame)

        # Display the frame with bounding boxes and labels
        cv2.imshow('Driver Behavior Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Ensure cleanup after the loop breaks
    print("Stopping camera and releasing resources...")
    picam2.stop()
    cv2.destroyAllWindows()

    # Release the video writer
    out.release()

    # Save the final emotion graph when exiting
    save_final_graph(emotion_counter)
