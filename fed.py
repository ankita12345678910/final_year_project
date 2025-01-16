import pyaudio
import numpy as np
import wave
import os
import cv2
from fer import FER
import pygame
import tkinter as tk
from tkinter import messagebox
import geocoder
from datetime import datetime
import mysql.connector
import time

# Set scream detection parameters
SAMPLE_RATE = 44100  # Sample rate for audio capture
CHUNK_SIZE = 1024  # Number of frames per buffer
THRESHOLD = 30  # Lower threshold for more sensitivity (test different values)
RECORDING_DURATION = 5  # Duration (in seconds) to record when a scream is detected
GAIN = 2  # Amplification for audio input

# Initialize the microphone stream for scream detection
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

# Folder to store recorded scream files
if not os.path.exists("captured_screams"):
    os.makedirs("captured_screams")

# Folder to store alert sound
if not os.path.exists("tune"):
    os.makedirs("tune")
# Initialize pygame mixer for sound (to play an alert)
pygame.mixer.init()

# Initialize the emotion detector
detector = FER()

# Load the pre-trained gender detection model for gender detection
gender_net = cv2.dnn.readNetFromCaffe(
    'deploy_gender2.prototxt',
    'gender_net.caffemodel'
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# List of gender labels
gender_list = ['Male', 'Female']

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        unconnected_out_layers = net.getUnconnectedOutLayers().flatten()
    except AttributeError:
        unconnected_out_layers = net.getUnconnectedOutLayers()
    return [layer_names[i - 1] for i in unconnected_out_layers]

# Function to connect to the database
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

# Function to store emotion data into the database
def store_emotion_data(latitude, longitude, emotion, gender, timestamp, image_path, audio_filename, weapon_detected):
    try:
        conn = connect_db()
        if conn is not None:
            cursor = conn.cursor()
            # Insert data into the table
            query = "INSERT INTO emotion_logs (latitude, longitude, emotion, gender, timestamp, image_path, audio, weapon_detected) VALUES (%s, %s, %s, %s, %s, %s, %s,%s)"
            values = (latitude, longitude, emotion, gender, timestamp, image_path, audio_filename, weapon_detected)  # Store the filename instead of binary data
            cursor.execute(query, values)
            conn.commit()
            print(f"Data saved to the database: {values}")  # Debugging line
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if conn:
            cursor.close()
            conn.close()


# Function to detect gender from the face
def detect_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0))
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    return gender

# Function to detect scream from the microphone input
def detect_scream():
    try:
        # Capture audio data
        audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)

        # Amplify the audio signal
        audio_data = audio_data * GAIN
        
        # Calculate the volume level (root mean square of audio data)
        volume = np.sqrt(np.mean(audio_data**2))

        print(f"Volume: {volume}")  # Debugging line to check the volume

        # If the volume exceeds the threshold, detect a scream
        if volume > THRESHOLD:
            print("Scream detected!")
            return True  # Scream detected
        else:
            return False  # No scream detected
    except Exception as e:
        print(f"Error in detecting scream: {e}")
        return False

def record_scream_frames():
    try:
        print("Recording scream...")
        frames = []
        start_time = time.time()
        
        while time.time() - start_time < RECORDING_DURATION:
            data = stream.read(CHUNK_SIZE)
            frames.append(data)
        
        return frames  # Return the captured frames
    
    except Exception as e:
        print(f"Error while recording scream: {e}")
        return None

def store_recorded_scream(frames):
    try:
        if frames is None or len(frames) == 0:
            print("No frames to save.")
            return None
        
        # Save the recorded scream to a .wav file with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"captured_screams/scream_{timestamp}.wav"
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono channel
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))  # 2 bytes for 16-bit audio
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"Scream recorded and saved to {filename}")
        
        # Return the filename, not the audio data
        return filename  # Return the filename of the saved audio
        
    except Exception as e:
        print(f"Error while storing recorded scream: {e}")
        return None


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

def show_alert(gender, emotion_label, image_path, audio_filename, weapon_detected,weapon_name=None):
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
        store_emotion_data(latitude, longitude, emotion_label, gender, current_time, image_path, audio_filename, weapon_detected)

        # Play the sound continuously in a loop
        sound_file_path = os.path.join("tune", "alert_tune.mp3")  # Path to the sound file inside the "tune" folder
        pygame.mixer.music.load(sound_file_path)  # Load the sound file
        pygame.mixer.music.play(-1)  # Loop the sound infinitely
        
        # Create a pop-up alert window
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        message = f"A {gender} in Danger.\n\nEmotion: {emotion_label}\nLocation: {latitude}, {longitude}\nDate/Time: {current_time}"
        
        if weapon_detected:
            message += f"\nWeapon Detected: {weapon_name}"
        else:
            message += "\nNo Weapon Detected"
        
        # Stop the sound after the user clicks OK
        pygame.mixer.music.stop()

        # Destroy the tkinter window after alert
        root.destroy()  
    except Exception as e:
        print(f"Error in show_alert: {e}")


def detect_weapon(frame):
    # Prepare the frame for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    weapon_detected = False
    weapon_name = None
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # If confidence is greater than 0.3 and the detected class is a weapon (e.g., gun, knife)
            if confidence > 0.3 and classes[class_id] in ['knife', 'gun']:  # Add more weapons if necessary
                weapon_detected = True
                weapon_name = classes[class_id]
                
                # Draw bounding box around detected weapon
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue border
                label = f"{weapon_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                print(f"Detected {label} at [{x}, {y}, {w}, {h}] with confidence {confidence:.2f}")
                
    return weapon_detected, weapon_name


# Main loop to detect scream and emotions
def main_loop():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect emotions in the current frame
        emotions = detector.detect_emotions(frame)
        
        # Default values for emotion and gender
        emotion_label = None
        gender = None
        
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

        # Check if 'fear' emotion and 'Female' gender are detected
        if emotion_label == 'fear' and gender == 'Female':
            print("Fear detected!")
            weapon_detected, weapon_name = detect_weapon(frame)  # Detect weapon
                
            if weapon_detected:
                cv2.putText(frame, "Fear and Weapon Detected!", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                show_alert(gender, emotion_label, save_image(frame, emotion_label), True, weapon_name)  # Pass weapon_name
            else:
                cv2.putText(frame, "Fear Detected!", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                show_alert(gender, emotion_label, save_image(frame, emotion_label), False, None)  # Pass None for weapon_name

            image_path = save_image(frame, emotion_label)  # Save the captured image

            # Store the recorded scream when fear is detected
            scream_frames = record_scream_frames()  # Capture scream audio frames
            audio_data = store_recorded_scream(scream_frames)  # Store the captured audio

            if audio_data:
                # Trigger the alert and store the data in the database
                show_alert(gender, emotion_label, image_path, audio_data, False, None)  # Add appropriate arguments here

        
        # Display the frame with emotion annotations
        cv2.imshow('Real-Time Emotion Detection', frame)
        
        # Check if 'q' is pressed to terminate the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()