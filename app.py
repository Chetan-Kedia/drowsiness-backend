from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import base64
import time
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("drowsiness_svm.pkl")
scaler = joblib.load("scaler.pkl")

IMG_SIZE = 32

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------- STATE VARIABLES --------

closed_counter = 0
open_counter = 0
blink_count = 0
fatigue_score = 0
drowsy_state = False
closed_start_time = None

THRESHOLD = 3
DROWSY_TIME = 2


# -------- ROOT ROUTE (needed for Render) --------

@app.route("/")
def home():
    return "Backend running"


# -------- PREDICT --------

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
            "eye_state": "-",
            "blink": blink_count,
            "fatigue": fatigue_score
        })

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]

    fh, fw = face.shape

    eye_region = face[int(fh*0.2):int(fh*0.5), int(fw*0.2):int(fw*0.8)]

    eye_region = cv2.resize(eye_region, (IMG_SIZE, IMG_SIZE))

    feature = eye_region.flatten().reshape(1, -1)

    feature = scaler.transform(feature)

    prediction = model.predict(feature)[0]

    # ---------- smoothing ----------

    if prediction == 1:

        closed_counter += 1
        open_counter = 0

        if closed_counter == 1:
            closed_start_time = time.time()

    else:

        if closed_counter > 0:
            blink_count += 1

        open_counter += 1
        closed_counter = 0
        closed_start_time = None

    if closed_counter >= THRESHOLD:
        drowsy_state = True

    if open_counter >= THRESHOLD:
        drowsy_state = False

    # ---------- fatigue ----------

    if closed_start_time is not None:

        elapsed = time.time() - closed_start_time

        if elapsed > DROWSY_TIME:
            fatigue_score += 1
            drowsy_state = True

    # ---------- status ----------

    if drowsy_state:
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
        "eye_state": eye_state,
        "blink": blink_count,
        "fatigue": fatigue_score
    })


# -------- RENDER RUN FIX --------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)