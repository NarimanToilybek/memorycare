# server.py
import io, os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# === ПУТЬ К ЛУЧШЕЙ МОДЕЛИ (epoch5) ===
from pathlib import Path
import os

# Папка где лежит server.py
BASE_DIR = Path(__file__).resolve().parent

# Куда сохраняем модель на Render
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILENAME = "models/alz_vgg16_epoch5.h5"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

# 1) В Render -> Environment добавишь переменную GDRIVE_MODEL_ID
GDRIVE_MODEL_ID = os.getenv("GDRIVE_MODEL_ID", "").strip()

def ensure_model():
    if MODEL_PATH.exists():
        return

    if not GDRIVE_MODEL_ID:
        raise RuntimeError("GDRIVE_MODEL_ID is not set, and model file is missing.")

    # gdown умеет качать большие файлы с Drive (с confirm-страницей)
    import gdown
    url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"
    gdown.download(url, str(MODEL_PATH), quiet=False)

# Скачиваем (если надо) и только потом грузим модель
ensure_model()
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)


# Порядок классов (как при обучении): алфавит по именам папок
CLASS_NAMES = ["mild_demented", "moderate_demented", "no_demented", "very_mild_demented"]
IMG_SIZE = (224, 224)

app = FastAPI()
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
