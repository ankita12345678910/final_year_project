import cv2
from fer import FER
import geocoder
import tkinter as tk
from tkinter import messagebox

# Initialize the emotion detector
detector = FER()

# Function to display alert with geographical location
def show_alert():
    # Get the geographical coordinates
    g = geocoder.ip('me')  # This fetches the current location based on the IP address
    latitude, longitude = g.latlng
    
    # Create a pop-up alert window
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    messagebox.showwarning("Fear Detected!", f"A person showing fear has been detected.\n\nLocation: {latitude}, {longitude}")
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
        
        # Trigger the alert if the emotion is "fear"
        if emotion_label == 'fear':
            show_alert()

    # Display the frame with emotion annotations
    cv2.imshow('Real-Time Emotion Detection', frame)
    
    # Check if 'q' is pressed to terminate the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
