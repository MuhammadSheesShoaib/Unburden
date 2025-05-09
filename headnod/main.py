from fastapi import FastAPI
import cv2
import mediapipe as mp
import numpy as np
from fastapi.responses import PlainTextResponse

# Initialize FastAPI
app = FastAPI()

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Define thresholds for movement in pixels
nod_threshold = 10
shake_threshold = 10
prev_y, prev_x = None, None
nod_detected, shake_detected = False, False

cap = cv2.VideoCapture(0)

def detect_head_movement(frame: np.ndarray, prev_coords):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            h, w, _ = frame.shape
            nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)

            if prev_coords["x"] is None or prev_coords["y"] is None:
                prev_coords["x"] = nose_x
                prev_coords["y"] = nose_y
                return "none", prev_coords

            dx = abs(nose_x - prev_coords["x"])
            dy = abs(nose_y - prev_coords["y"])

            prev_coords["x"] = nose_x
            prev_coords["y"] = nose_y

            if dy > nod_threshold:
                return "nod", prev_coords
            elif dx > shake_threshold:
                return "shake", prev_coords
            else:
                return "none", prev_coords
    return "none", prev_coords


@app.post("/detect_head_movement/")
async def detect_head_movement_api():
    start_time = cv2.getTickCount()

    duration = 3.5  # Hardcoded
    prev_coords = {"x": None, "y": None}
    nod_motion_sum = 0
    shake_motion_sum = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"error": "Failed to capture frame"}

        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time > duration:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_tip = face_landmarks.landmark[1]
                h, w, _ = frame.shape
                nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)

                if prev_coords["x"] is None or prev_coords["y"] is None:
                    prev_coords["x"] = nose_x
                    prev_coords["y"] = nose_y
                    continue

                dx = abs(nose_x - prev_coords["x"])
                dy = abs(nose_y - prev_coords["y"])

                shake_motion_sum += dx
                nod_motion_sum += dy

                prev_coords["x"] = nose_x
                prev_coords["y"] = nose_y

        cv2.waitKey(1)

    # Final decision
    if nod_motion_sum > shake_motion_sum and nod_motion_sum > 15:
        movement = "nod"
    elif shake_motion_sum > nod_motion_sum and shake_motion_sum > 15:
        movement = "shake"
    else:
        movement = "none"

    # Return plain text response
    return PlainTextResponse(content=movement)
