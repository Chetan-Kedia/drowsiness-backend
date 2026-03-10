from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import base64
import time
import os
import gdown

app = Flask(__name__)
CORS(app)

MODEL_PATH = "drowsiness_svm.pkl"
SCALER_PATH = "scaler.pkl"

MODEL_URL = "https://drive.google.com/uc?id=1qsX30X3c31yEKRRF9GRLCwTUofYTzAfl"
SCALER_URL = "https://drive.google.com/uc?id=1mX7pdCdBaCLwalMXkmtNvNaE_sPM_-Fr"


# ---------- Download files if not exist ----------

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(SCALER_PATH):
    print("Downloading scaler...")
    gdown.download(SCALER_URL, SCALER_PATH, quiet=False)


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

IMG_SIZE = 64


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


closed_counter = 0
open_counter = 0
blink_count = 0
fatigue_score = 0
drowsy_state = False
closed_start_time = None

THRESHOLD = 3
DROWSY_TIME = 2


@app.route("/")
def home():
    return "Backend running"


@app.route("/predict", methods=["POST"])
def predict():

    global closed_counter, open_counter
    global blink_count, fatigue_score
    global drowsy_state, closed_start_time

    data = request.json["image"]

    img_bytes = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({
            "status": "No Face",
            "confidence": 0,
            "eye_state": "-"
        })

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]

    fh, fw = face.shape

    eye_region = face[int(fh*0.2):int(fh*0.5), int(fw*0.2):int(fw*0.8)]

    eye_region = cv2.resize(eye_region, (IMG_SIZE, IMG_SIZE))

    feature = eye_region.flatten().reshape(1, -1)

    feature = scaler.transform(feature)

    prediction = model.predict(feature)[0]

    if prediction == 1:
        status = "Drowsy"
        eye_state = "Closed"
        confidence = 92
    else:
        status = "Alert"
        eye_state = "Open"
        confidence = 96

    return jsonify({
        "status": status,
        "confidence": confidence,
        "eye_state": eye_state
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)