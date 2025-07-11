import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer

# Constants
ALARM_THRESHOLD = 10

# Load alarm sound
mixer.init()
sound = mixer.Sound("alarm.wav")

# Load trained eye model
model = load_model("models/eye_model.keras")

# Haar cascades
face_cascade = cv2.CascadeClassifier("haar cascade files/haarcascade_frontalface_alt.xml")
left_eye_cascade = cv2.CascadeClassifier("haar cascade files/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("haar cascade files/haarcascade_righteye_2splits.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    right_eye_status = 1
    left_eye_status = 1

    for (x, y, w, h) in faces:
        face_roi_gray = gray[y:y+h, x:x+w]

        right_eyes = right_eye_cascade.detectMultiScale(face_roi_gray)
        left_eyes = left_eye_cascade.detectMultiScale(face_roi_gray)

        for (ex, ey, ew, eh) in right_eyes:
            eye = face_roi_gray[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (24, 24)) / 255.0
            eye = eye.reshape(1, 24, 24, 1)
            pred = model.predict(eye, verbose=0)
            right_eye_status = np.argmax(pred)
            break

        for (ex, ey, ew, eh) in left_eyes:
            eye = face_roi_gray[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (24, 24)) / 255.0
            eye = eye.reshape(1, 24, 24, 1)
            pred = model.predict(eye, verbose=0)
            left_eye_status = np.argmax(pred)
            break

        break

    if right_eye_status == 0 and left_eye_status == 0:
        score += 1
        cv2.putText(frame, "Eyes Closed", (10, 30), font, 1, (0, 0, 255), 2)
    else:
        score -= 1
        cv2.putText(frame, "Eyes Open", (10, 30), font, 1, (0, 255, 0), 2)

    score = max(0, score)

    if score > ALARM_THRESHOLD:
        if not mixer.get_busy():
            sound.play()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

    cv2.putText(frame, f"Score: {score}", (300, 30), font, 1, (0, 0, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
