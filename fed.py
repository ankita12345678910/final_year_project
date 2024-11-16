import cv2
from fer import FER
import geocoder
import tkinter as tk
from tkinter import messagebox
from datetime import datetime

# Initialize the emotion detector
detector = FER()

# Load the pre-trained gender detection model
gender_net = cv2.dnn.readNetFromCaffe(
    'deploy_gender2.prototxt',
    'gender_net.caffemodel'
)

# List of gender labels
gender_list = ['Male', 'Female']

# Function to detect gender
def detect_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0))
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    return gender

# Function to display alert with geographical location, gender, and date/time
def show_alert(gender):
    # Get the geographical coordinates
    g = geocoder.ip('me')  # This fetches the current location based on the IP address
    latitude, longitude = g.latlng
    
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a pop-up alert window
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    messagebox.showwarning("Fear Detected!", f"A Woman in Danger.\n\nLocation: {latitude}, {longitude}\nDate/Time: {current_time}")
    root.destroy()  # Close the tkinter window after alert

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break  # Exit loop if the frame was not captured correctly
    
    # Detect emotions in the current frame
    emotions = detector.detect_emotions(frame)
    
    for emotion in emotions:
        # Extract bounding box and draw it on the frame
        x, y, w, h = emotion['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Find the emotion with the highest score
        emotion_label = max(emotion['emotions'], key=emotion['emotions'].get)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Detect gender
        face = frame[y:y+h, x:x+w]
        gender = detect_gender(face)
        cv2.putText(frame, gender, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Trigger the alert if the emotion is "fear" and the gender is "Female"
        if emotion_label == 'fear' and gender == 'Female':
            cv2.putText(frame, "Fear Detected!", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Real-Time Emotion Detection', frame)
            cv2.waitKey(1000)  # Display the tag for 1 second before the alert
            show_alert(gender)

    # Display the frame with emotion annotations
    cv2.imshow('Real-Time Emotion Detection', frame)
    
    # Check if 'q' is pressed to terminate the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
