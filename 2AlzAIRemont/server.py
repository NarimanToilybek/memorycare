from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import os, io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import gdown

app = FastAPI()

# ===== STATIC SITE =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# ===== MODEL =====
MODEL_NAME = "alz_vgg16_epoch5.h5"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
MODEL_URL = "https://drive.google.com/uc?id=ТВОЙ_ID"

os.makedirs(MODELS_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    "mild_demented",
    "moderate_demented",
    "no_demented",
    "very_mild_demented"
]

IMG_SIZE = (224, 224)

def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)[None, ...]
    return preprocess_input(x)

# ===== API =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    x = preprocess(img)

    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))

    return {
        "label": CLASS_NAMES[idx],
        "prob": float(preds[idx]),
        "probs": {CLASS_NAMES[i]: float(preds[i]) for i in range(4)}
    }
