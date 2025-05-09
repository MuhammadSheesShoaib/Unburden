import os
os.environ["KERAS_BACKEND"] = "jax"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import keras
import numpy as np
import cv2
from PIL import Image
import io
from huggingface_hub import hf_hub_download

app = FastAPI()

# Add root route for testing
@app.get("/")
def read_root():
    return {"message": "Facial Emotion API is running 🚀"}

# Load model and config
model = None
desired_emotions = ['happy', 'sad', 'neutral']
original_emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
desired_indices = [original_emotion_labels.index(emotion) for emotion in desired_emotions]

@app.on_event("startup")
def load_emotion_model():
    global model
    print("🔄 Downloading model from HuggingFace Hub...")
    model_path = hf_hub_download(
    repo_id="Shees7/facial_7_cat",
    filename="trained_model.keras",
    cache_dir="/tmp"
    )
    print("✅ Model file downloaded at:", model_path)
    model = keras.saving.load_model(model_path)
    print("✅ Model loaded successfully.")

def preprocess_face(image_bytes):
    np_img = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = np_img[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (224, 224))
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=0)
    return face_expanded

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={"error": "Model not loaded."}, status_code=500)

    image_bytes = await file.read()
    processed_face = preprocess_face(image_bytes)

    if processed_face is None:
        return {"emotion": "neutral"}

    predictions = model.predict(processed_face)[0]
    filtered_predictions = [predictions[i] for i in desired_indices]
    predicted_index = np.argmax(filtered_predictions)
    predicted_emotion = desired_emotions[predicted_index]

    return {"emotion": predicted_emotion}
