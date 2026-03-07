from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import base64
import os

app = Flask(__name__)
CORS(app)


# ================= LOAD MODEL =================

model = joblib.load("drowsiness_svm.pkl")
scaler = joblib.load("scaler.pkl")

IMG_SIZE = 32


# ================= ROOT ROUTE =================

@app.route("/")
def home():
    return "Backend running"


# ================= PREDICT ROUTE =================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    # decode base64 image
    img_bytes = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    gray = gray.flatten().reshape(1, -1)

    gray = scaler.transform(gray)

    prediction = model.predict(gray)[0]

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


# ================= RUN FOR RENDER =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)