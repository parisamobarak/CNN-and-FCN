import os
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from keras import mixed_precision
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image
import io

router = APIRouter()

# تنظیمات مربوط به پردازش داده‌ها و آموزش مدل
AUTOTUNE = tf.data.AUTOTUNE  # استفاده بهینه از پردازنده برای پردازش داده‌ها

NUM_CLASSES = 4  # تعداد کلاس‌ها در مجموعه داده
INPUT_HEIGHT = 224  # ارتفاع ورودی تصویر
INPUT_WIDTH = 224  # عرض ورودی تصویر
LEARNING_RATE = 1e-3  # نرخ یادگیری برای به‌روزرسانی وزن‌ها
WEIGHT_DECAY = 1e-4  # کاهش وزن برای جلوگیری از overfitting
EPOCHS = 10  # تعداد اپوک‌ها برای آموزش مدل
BATCH_SIZE = 32  # اندازه هر بچ در آموزش
MIXED_PRECISION = True  # استفاده از پردازش دقیق مخلوط (Mixed Precision) برای سرعت بیشتر آموزش
SHUFFLE = True  # مخلوط کردن داده‌ها قبل از آموزش

#تنظیمات( Mixed-Precision) برای آموزش سریعتر و با مصرف حافظه کمتر برای مدل
if MIXED_PRECISION:
    policy = mixed_precision.Policy("mixed_float16")  # تعریف سیاست Mixed Precision
    mixed_precision.set_global_policy(policy)  # تنظیم سیاست به صورت سراسری

def train_and_save_model():
    # بارگذاری مجموعه داده و تقسیم آن به مجموعه‌های آموزشی، اعتبارسنجی و آزمایشی
    (train_ds, valid_ds, test_ds) = tfds.load(
        "oxford_iiit_pet",  # نام مجموعه داده
        split=["train[:85%]", "train[85%:]", "test"],
        batch_size=BATCH_SIZE,  # اندازه هر بچ
        shuffle_files=SHUFFLE,  # مخلوط کردن داده‌ها
    )

    #تغییر اندازه تصاویر و ماسک‌های بخش‌بندی به ابعاد ثابت (224x224)
    def unpack_resize_data(section):
        image = section["image"]  # تصویر
        segmentation_mask = section["segmentation_mask"]  # ماسک بخش‌بندی

        resize_layer = tf.keras.layers.Resizing(INPUT_HEIGHT, INPUT_WIDTH)  # لایه تغییر اندازه به 224x224

        image = resize_layer(image)  # تغییر اندازه تصویر
        segmentation_mask = resize_layer(segmentation_mask)  # تغییر اندازه ماسک

        return image, segmentation_mask

    # اعمال تابع تغییر اندازه بر روی تصاویر و ماسک‌ها در مجموعه داده‌های آموزشی، اعتبارسنجی و آزمایشی
    train_ds = train_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
    valid_ds = valid_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)


     #پیش‌پردازش تصاویر ورودی برای استفاده در مدل VGG19
    def preprocess_data(image, segmentation_mask):
        image = tf.keras.applications.vgg19.preprocess_input(image)  # پیش‌پردازش تصویر برای مدل VGG19

        return image, segmentation_mask

    # پیش‌پردازش داده‌ها، مخلوط کردن و پیش‌پردازش داده‌های آموزشی، اعتبارسنجی و آزمایشی
    train_ds = (
        train_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size=1024)  # مخلوط کردن داده‌ها
        .prefetch(buffer_size=AUTOTUNE)  # پیش‌پردازش داده‌ها
    )
    valid_ds = (
        valid_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size=1024)
        .prefetch(buffer_size=AUTOTUNE)
    )
    test_ds = (
        test_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size=1024)
        .prefetch(buffer_size=AUTOTUNE)
    )
    #تعریف لایه ورودی مدل با ابعاد 224x224x3
    input_layer = tf.keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))

    # بارگذاری مدل VGG19 از پیش آموزش‌دیده شده با وزن‌های ImageNet
    vgg_model = tf.keras.applications.VGG19(include_top=True, weights="imagenet")

    # تعریف مدل FCN Backbone با استفاده از لایه‌های میانی VGG19
    fcn_backbone = tf.keras.models.Model(
        inputs=vgg_model.layers[1].input,
        outputs=[
            vgg_model.get_layer(block_name).output
            for block_name in ["block3_pool", "block4_pool", "block5_pool"]
        ],
    )

    # غیرقابل آموزش کردن مدل FCN Backbone برای استفاده از وزن‌های از پیش آموزش داده شده
    fcn_backbone.trainable = False

    #استخراج ویژگی‌ها از مدل FCN Backbone با استفاده از لایه ورودی
    x = fcn_backbone(input_layer)

    # تبدیل لایه‌های Dense به لایه‌های Conv2D
    units = [4096, 4096]  # تعداد فیلترها در لایه‌های Dense
    dense_convs = []

    for filter_idx in range(len(units)):
        dense_conv = tf.keras.layers.Conv2D(
            filters=units[filter_idx],  # تعداد فیلترها
            kernel_size=(7, 7) if filter_idx == 0 else (1, 1),  # اندازه کرنل
            strides=(1, 1),  # اندازه گام
            activation="relu",  # تابع فعال‌سازی
            padding="same",  # اضافه کردن صفر در اطراف تصویر
            use_bias=False,  # عدم استفاده از بایاس
            kernel_initializer=tf.keras.initializers.Constant(1.0),  # مقداردهی اولیه وزن‌ها
        )
        dense_convs.append(dense_conv)
        dropout_layer = tf.keras.layers.Dropout(0.5)  # لایه Dropout برای جلوگیری از Overfitting
        dense_convs.append(dropout_layer)

    dense_convs = tf.keras.Sequential(dense_convs)  # مدل Sequential برای لایه‌های Dense و Dropout
    dense_convs.trainable = False  # غیر قابل آموزش کردن لایه‌های Dense

    x[-1] = dense_convs(x[-1])  # اعمال لایه‌های Dense و Dropout بر روی خروجی آخر

    pool3_output, pool4_output, pool5_output = x  # گرفتن خروجی‌های لایه‌های مختلف

    pool5 = tf.keras.layers.Conv2D(  # لایه Conv2D برای تنظیم تعداد کانال‌ها به تعداد کلاس‌ها
        filters=NUM_CLASSES,
        kernel_size=(1, 1),
        padding="same",
        strides=(1, 1),
        activation="relu",
    )

    # ایجاد لایه Softmax برای پیش‌بینی کلاس‌های مختلف
    fcn32s_conv_layer = tf.keras.layers.Conv2D(
        filters=NUM_CLASSES,
        kernel_size=(1, 1),
        activation="softmax",
        padding="same",
        strides=(1, 1),
    )

    # افزایش مقیاس تصویر به اندازه اصلی با لایه UpSampling2D
    fcn32s_upsampling = tf.keras.layers.UpSampling2D(
        size=(32, 32),  # اندازه افزایش مقیاس
        data_format=tf.keras.backend.image_data_format(),
        interpolation="bilinear",  # نوع اینترپولیشن
    )

    final_fcn32s_pool = pool5(pool5_output)  # اعمال لایه Conv2D برای خروجی نهایی
    final_fcn32s_output = fcn32s_conv_layer(final_fcn32s_pool)  # اعمال لایه Softmax
    final_fcn32s_output = fcn32s_upsampling(final_fcn32s_output)  # افزایش مقیاس خروجی به اندازه تصویر ورودی

    fcn32s_model = tf.keras.Model(inputs=input_layer, outputs=final_fcn32s_output)  # ساخت مدل FCN32s

    # تنظیم وزن‌های لایه‌های Dense از مدل VGG19 برای مدل FCN32s
    weights1 = vgg_model.get_layer("fc1").get_weights()[0]
    weights2 = vgg_model.get_layer("fc2").get_weights()[0]

    weights1 = weights1.reshape(7, 7, 512, 4096)  # تغییر اندازه وزن‌های fc1
    weights2 = weights2.reshape(1, 1, 4096, 4096)  # تغییر اندازه وزن‌های fc2

    dense_convs.layers[0].set_weights([weights1])  # تنظیم وزن‌های لایه اول Dense
    dense_convs.layers[2].set_weights([weights2])  # تنظیم وزن‌های لایه دوم Dense

    # تعریف بهینه‌ساز AdamW با نرخ یادگیری و کاهش وزن
    fcn32s_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    fcn32s_loss = tf.keras.losses.SparseCategoricalCrossentropy()  # تعریف تابع خسارت برای دسته‌بندی

    #کامپاسل کردن مدل
    fcn32s_model.compile(
        optimizer=fcn32s_optimizer,
        loss=fcn32s_loss,
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),  # معیار mIOU
            tf.keras.metrics.SparseCategoricalAccuracy(),  # معیار دقت
        ],
    )

    #آموز مدل
    fcn32s_model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds)  # آموزش مدل

    # ذخیره مدل آموزش داده شده
    fcn32s_model.save("fcn32s_model.h5")

# بارگذاری مدل آموزش داده شده
def load_and_predict_model():
    fcn32s_model = load_model("fcn32s_model.h5")
    return fcn32s_model

# این بخش فقط یک بار برای آموزش و ذخیره مدل اجرا می‌شود.
if not os.path.exists("fcn32s_model.h5"):
    train_and_save_model()

# بارگذاری مدل برای پیش‌بینی
fcn32s_model = load_and_predict_model()


# پیش‌پردازش تصویر ورودی
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # تغییر اندازه تصویر به 224x224
    image = tf.keras.applications.vgg19.preprocess_input(image)  # پیش‌پردازش برای مدل VGG19
    image = np.expand_dims(image, axis=0)  # اضافه کردن بعد Batch
    return image


# پیش‌بینی ماسک بخش بندی با استفاده از مدل
def predict_mask(image, model):
    pred_mask = model.predict(image).astype("float")  # پیش‌بینی ماسک
    pred_mask = np.argmax(pred_mask, axis=-1)  # انتخاب کلاس با بالاترین احتمال
    pred_mask = pred_mask[0, ...]  # برداشتن بعد Batch
    return pred_mask #  بازگشت ماسک پیش‌بینی شده بصورت ماتریس


#تبدیل ماسک بخش بندی به تصویر
def mask_to_image(mask, colormap=plt.cm.viridis):
    # تبدیل ماسک به تصویر رنگی
    normed_mask = mask / np.max(mask)  # نرمال‌سازی ماسک
    colormap_mask = colormap(normed_mask)  # اعمال colormap
    image_array = (colormap_mask[:, :, :3] * 255).astype(np.uint8)  # تبدیل به uint8 برای ساختن تصویر
    return Image.fromarray(image_array)

@router.post("/predictfcn/")
async def predict(file: UploadFile = File(...)):
    # مسیر API برای پیش‌بینی ماسک بر اساس تصویر ورودی
    image = np.frombuffer(await file.read(), np.uint8)  # خواندن تصویر از فایل ورودی
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # تبدیل بایت‌ها به تصویر
    preprocessed_image = preprocess_image(image)  # پیش‌پردازش تصویر

    mask_32s = predict_mask(preprocessed_image, fcn32s_model)  # پیش‌بینی ماسک

    mask_image = mask_to_image(mask_32s) # تبدیل ماسک به تصویر

    # ذخیره تصویر در حافظه به عنوان بایت‌ها
    img_byte_arr = io.BytesIO()
    mask_image.save(img_byte_arr, format='PNG')  # ذخیره تصویر به فرمت PNG
    img_byte_arr.seek(0)  # بازنشانی موقعیت به ابتدای بایت‌ها

    return StreamingResponse(img_byte_arr, media_type="image/png")  # ارسال تصویر به عنوان پاسخ
