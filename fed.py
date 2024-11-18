import cv2
from fer import FER
import geocoder
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import mysql.connector
import os

# Initialize the emotion detector
detector = FER()

# Load the pre-trained gender detection model
gender_net = cv2.dnn.readNetFromCaffe(
    'deploy_gender2.prototxt',
    'gender_net.caffemodel'
)

# List of gender labels
gender_list = ['Male', 'Female']

def connect_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="womensafety",
            charset="utf8mb4",
            collation="utf8mb4_general_ci"
        )
        print("Database connection successful!")
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def store_emotion_data(latitude, longitude, emotion, gender, timestamp, image_path):
    try:
        conn = connect_db()
        if conn is not None:
            cursor = conn.cursor()
            # Insert data into the table
            query = "INSERT INTO emotion_logs (latitude, longitude, emotion, gender, timestamp, image_path) VALUES (%s, %s, %s, %s, %s, %s)"
            values = (latitude, longitude, emotion, gender, timestamp, image_path)
            cursor.execute(query, values)
            conn.commit()
            print(f"Data saved to the database: {values}")  # Debugging line
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# Function to detect gender
def detect_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0))
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    return gender

# Function to display alert with geographical location, gender, and date/time
def show_alert(gender, emotion_label, image_path):
    try:
        # Get the geographical coordinates
        g = geocoder.ip('me')  # This fetches the current location based on the IP address
        latitude, longitude = g.latlng
        
        # Check if we got the location successfully
        if latitude is None or longitude is None:
            print("Failed to get the location!")
            return
        
        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store the emotion data in the database
        store_emotion_data(latitude, longitude, emotion_label, gender, current_time, image_path)
        
        # Create a pop-up alert window
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        messagebox.showwarning("Fear Detected!", f"A {gender} in Danger.\n\nEmotion: {emotion_label}\nLocation: {latitude}, {longitude}\nDate/Time: {current_time}")
        root.destroy()  # Close the tkinter window after alert
    except Exception as e:
        print(f"Error in show_alert: {e}")

# Function to save captured images
def save_image(frame, emotion_label):
    # Create a directory to store images if it doesn't exist
    if not os.path.exists("captured_images"):
        os.makedirs("captured_images")
    
    # Generate a unique file name based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"captured_images/{emotion_label}_{timestamp}.jpg"
    
    # Save the image
    cv2.imwrite(image_path, frame)
    return image_path

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
            image_path = save_image(frame, emotion_label)  # Save the captured image
            show_alert(gender, emotion_label, image_path)  # This will store data and show alert

    # Display the frame with emotion annotations
    cv2.imshow('Real-Time Emotion Detection', frame)
    
    # Check if 'q' is pressed to terminate the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
