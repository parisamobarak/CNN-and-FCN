import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import uvicorn
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os
from PIL import Image
from fastapi import APIRouter

# تغییر انکدینگ خروجی به UTF-8
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


router = APIRouter()

# نام کلاس‌ها برای مجموعه داده CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def train_and_save_model():
    # بارگذاری و نرمال‌سازی داده‌ها
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # تعریف مدل
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # کامپایل کردن مدل
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # آموزش مدل
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    # ذخیره مدل به فرمت .keras
    model.save('cifar10_model.keras')


def load_and_predict_model():
    # بارگذاری مدل آموزش داده شده
    model = load_model('cifar10_model.keras')
    return model


# این بخش فقط یک بار برای آموزش و ذخیره مدل اجرا می‌شود.
if not os.path.exists('cifar10_model.keras'):
    train_and_save_model()

# بارگذاری مدل برای پیش‌بینی
model = load_and_predict_model()


# تابع برای پردازش تصویر بارگذاری‌شده
def process_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# تعریف مسیر پیش‌بینی
@router.post("/predictcnn/")
async def predict(file: UploadFile = File(...)):
    # خواندن فایل تصویری بارگذاری‌شده
    contents = await file.read()
    # پردازش تصویر
    img_array = process_image(contents)
    # انجام پیش‌بینی
    predictions = model.predict(img_array)
    score = np.argmax(predictions[0])
    # دریافت برچسب
    predicted_label = class_names[score]
    return {"label": predicted_label}



