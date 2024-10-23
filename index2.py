from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

# Function to use text-to-speech
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Open the webcam
video = cv2.VideoCapture(0)

# Load the face detection classifier
facedetect_path = 'data/haarcascade_frontalface_default.xml'
if not os.path.exists(facedetect_path):
    raise Exception(f"Haar cascade file not found at {facedetect_path}")
facedetect = cv2.CascadeClassifier(facedetect_path)

# Load the labels and face data
labels_file_path = 'data/names.pkl'
faces_file_path = 'data/faces_data.pkl'

if not os.path.exists(labels_file_path) or not os.path.exists(faces_file_path):
    raise Exception("Label or face data file not found.")

with open(labels_file_path, 'rb') as w:
    LABELS = pickle.load(w)

with open(faces_file_path, 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image if it exists
background_image_path = "data/background.png"
if os.path.exists(background_image_path):
    imgBackground = cv2.imread(background_image_path)
else:
    imgBackground = np.zeros((480, 640, 3), dtype=np.uint8)  # Create a black background if not found

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    if not ret:  # Check if the frame is captured successfully
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Make sure to check the prediction output
        output = knn.predict(resized_img)
        if output.size == 0:
            print("No prediction made.")
            continue

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        # Create attendance CSV file if it doesn't exist
        attendance_file_path = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(attendance_file_path)

        # Draw rectangles and put text on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Prepare attendance entry
        attendance = [str(output[0]), str(timestamp)]

    imgBackground = frame
    cv2.imshow("Frame", imgBackground)

    k = cv2.waitKey(1)
    if k == ord('p'):
        speak("Attendance Taken..")
        time.sleep(5)

        with open(attendance_file_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)  # Write header only if file doesn't exist
            writer.writerow(attendance)  # Write attendance

    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
