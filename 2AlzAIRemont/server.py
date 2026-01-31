# server.py

import os
import gdown

import io, os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# === ПУТЬ К ЛУЧШЕЙ МОДЕЛИ (epoch5) ===
MODEL_DIR = "models"
MODEL_NAME = "alz_vgg16_epoch5.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

MODEL_URL = "https://drive.google.com/uc?id=1Gpn4Aw-Hji0gVlvSZECYD15NoVze63Sj"

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

ensure_model()
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# Порядок классов (как при обучении): алфавит по именам папок
CLASS_NAMES = ["mild_demented", "moderate_demented", "no_demented", "very_mild_demented"]
IMG_SIZE = (224, 224)

app = FastAPI()
# Раздаём статические файлы (css/js/png/jpg)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Если у тебя style.css и script.js лежат в корне 2AlzAIRemont:
# то их можно отдавать как есть (FastAPI сам не раздаёт файлы из корня),
# поэтому проще тоже смонтировать корень как static:
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def home():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def _preprocess_pil(img_pil: Image.Image) -> np.ndarray:
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)[np.newaxis, ...]
    x = preprocess_input(x)
    return x

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext == ".dcm":
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(raw))
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0: arr /= arr.max()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        x = _preprocess_pil(img)
    else:
        img = Image.open(io.BytesIO(raw))
        x = _preprocess_pil(img)

    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    probs = {n: float(p) for n, p in zip(CLASS_NAMES, preds)}
    ru = {
        "no_demented": "No Demented (Нет признаков деменции)",
        "very_mild_demented": "Very Mild Demented (Очень лёгкая деменция)",
        "mild_demented": "Mild Demented (Лёгкая деменция)",
        "moderate_demented": "Moderate Demented (Умеренная деменция)",
    }
    return {"label": label, "label_ru": ru[label], "prob": float(preds[idx]), "probs": probs}
