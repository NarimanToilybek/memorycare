import os
import re
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

# ===== 1) Пути =====
# Скрипт лежит в 2AlzAIRemont/ai → BASE_DIR = 2AlzAIRemont, а train там же
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
CKPT_DIR  = BASE_DIR  # сюда сохраняем/ищем *.h5

print("TRAIN_DIR =", TRAIN_DIR)
if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Папка {TRAIN_DIR} не найдена!")

# ===== 2) Настройки =====
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
SEED       = 42
WARMUP_EPOCHS   = 5      # Stage 1
FINETUNE_EPOCHS = 15     # Stage 2

# ===== 3) Датасеты =====
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Классы:", class_names)

# ===== 4) Предобработка =====
def preprocess(image, label):
    return preprocess_input(image), label

train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# ===== 5) Базовая модель (создаём свежую) =====
base = VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base.trainable = False  # сначала заморожена

model = models.Sequential([
    base,
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
])

# ===== 6) Callback: сохраняем модель после каждой эпохи =====
class SaveEachEpoch(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # epoch — глобальный (учитывает initial_epoch), поэтому имена продолжаются корректно
        filename = os.path.join(CKPT_DIR, f"alz_vgg16_epoch{epoch+1}.h5")
        self.model.save(filename)
        print(f">>> Модель сохранена: {filename}")

# ===== 7) Функции для чекпоинтов =====
def find_last_checkpoint(ckpt_dir: str) -> tuple[str | None, int]:
    """Возвращает (путь_к_последнему_файлу, номер_эпохи) либо (None, 0)."""
    files = glob.glob(os.path.join(ckpt_dir, "alz_vgg16_epoch*.h5"))
    if not files:
        return None, 0
    # Берём самый свежий по времени модификации
    files.sort(key=os.path.getmtime)
    last = files[-1]
    m = re.search(r"epoch(\d+)\.h5$", os.path.basename(last))
    epoch_num = int(m.group(1)) if m else 0
    return last, epoch_num

last_ckpt, last_epoch = find_last_checkpoint(CKPT_DIR)
print("Найден чекпоинт:", last_ckpt, "эпоха:", last_epoch)

# ===== 8) Логика обучения =====
if last_ckpt is None:
    # --- Чекпоинтов нет → полный цикл: warmup(5) + finetune(15)
    print("\n=== Stage 1: Заморозка base (5 эпох) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=[SaveEachEpoch()]
    )

    print("\n=== Stage 2: Fine-tuning (15 эпох) ===")
    # Размораживаем верхние слои VGG16 (пример: последние 4 conv-блока)
    for layer in base.layers[:-4]:
        layer.trainable = False
    for layer in base.layers[-4:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=[SaveEachEpoch()]
    )

else:
    # --- Чекпоинт есть → продолжаем FINE-TUNING c него
    print("\n>>> Восстанавливаем модель из чекпоинта и продолжаем fine-tuning")
    model = load_model(last_ckpt)

    # На всякий случай убеждаемся, что базовая VGG16 разморожена нужным образом.
    # В загруженной модели base — это нулевой слой Sequential.
    try:
        loaded_base = model.layers[0]
        # если это VGG16, откроем верхние слои
        if isinstance(loaded_base, tf.keras.Model) and loaded_base.name.startswith("vgg16"):
            for layer in loaded_base.layers[:-4]:
                layer.trainable = False
            for layer in loaded_base.layers[-4:]:
                layer.trainable = True
    except Exception:
        pass

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Продолжаем с корректной нумерацией эпох:
    # initial_epoch = номер последней завершённой эпохи
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,      # целевое значение (15) — Keras сам учтёт initial_epoch
        initial_epoch=min(last_epoch, FINETUNE_EPOCHS),  # не превысим 15
        callbacks=[SaveEachEpoch()]
    )

# ===== 9) Финальная модель =====
final_path = os.path.join(CKPT_DIR, "alz_vgg16_final.h5")
model.save(final_path)
print("Модель сохранена:", final_path)
