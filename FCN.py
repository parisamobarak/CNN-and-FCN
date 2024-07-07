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

AUTOTUNE = tf.data.AUTOTUNE

NUM_CLASSES = 4
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 10
BATCH_SIZE = 32
MIXED_PRECISION = True
SHUFFLE = True

# Mixed-precision setting
if MIXED_PRECISION:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

def train_and_save_model():
    (train_ds, valid_ds, test_ds) = tfds.load(
        "oxford_iiit_pet",
        split=["train[:85%]", "train[85%:]", "test"],
        batch_size=BATCH_SIZE,
        shuffle_files=SHUFFLE,
    )

    # Image and Mask Pre-processing
    def unpack_resize_data(section):
        image = section["image"]
        segmentation_mask = section["segmentation_mask"]

        resize_layer = tf.keras.layers.Resizing(INPUT_HEIGHT, INPUT_WIDTH)

        image = resize_layer(image)
        segmentation_mask = resize_layer(segmentation_mask)

        return image, segmentation_mask

    train_ds = train_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
    valid_ds = valid_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)

    def preprocess_data(image, segmentation_mask):
        image = tf.keras.applications.vgg19.preprocess_input(image)

        return image, segmentation_mask

    train_ds = (
        train_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size=1024)
        .prefetch(buffer_size=AUTOTUNE)
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

    input_layer = tf.keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))

    # VGG Model backbone with pre-trained ImageNet weights.
    vgg_model = tf.keras.applications.VGG19(include_top=True, weights="imagenet")

    # Extracting different outputs from same model
    fcn_backbone = tf.keras.models.Model(
        inputs=vgg_model.layers[1].input,
        outputs=[
            vgg_model.get_layer(block_name).output
            for block_name in ["block3_pool", "block4_pool", "block5_pool"]
        ],
    )

    # Setting backbone to be non-trainable
    fcn_backbone.trainable = False

    x = fcn_backbone(input_layer)

    # Converting Dense layers to Conv2D layers
    units = [4096, 4096]
    dense_convs = []

    for filter_idx in range(len(units)):
        dense_conv = tf.keras.layers.Conv2D(
            filters=units[filter_idx],
            kernel_size=(7, 7) if filter_idx == 0 else (1, 1),
            strides=(1, 1),
            activation="relu",
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(1.0),
        )
        dense_convs.append(dense_conv)
        dropout_layer = tf.keras.layers.Dropout(0.5)
        dense_convs.append(dropout_layer)

    dense_convs = tf.keras.Sequential(dense_convs)
    dense_convs.trainable = False

    x[-1] = dense_convs(x[-1])

    pool3_output, pool4_output, pool5_output = x

    # 1x1 convolution to set channels = number of classes
    pool5 = tf.keras.layers.Conv2D(
        filters=NUM_CLASSES,
        kernel_size=(1, 1),
        padding="same",
        strides=(1, 1),
        activation="relu",
    )

    # Get Softmax outputs for all classes
    fcn32s_conv_layer = tf.keras.layers.Conv2D(
        filters=NUM_CLASSES,
        kernel_size=(1, 1),
        activation="softmax",
        padding="same",
        strides=(1, 1),
    )

    # Up-sample to original image size
    fcn32s_upsampling = tf.keras.layers.UpSampling2D(
        size=(32, 32),
        data_format=tf.keras.backend.image_data_format(),
        interpolation="bilinear",
    )

    final_fcn32s_pool = pool5(pool5_output)
    final_fcn32s_output = fcn32s_conv_layer(final_fcn32s_pool)
    final_fcn32s_output = fcn32s_upsampling(final_fcn32s_output)

    fcn32s_model = tf.keras.Model(inputs=input_layer, outputs=final_fcn32s_output)

    # VGG's last 2 layers
    weights1 = vgg_model.get_layer("fc1").get_weights()[0]
    weights2 = vgg_model.get_layer("fc2").get_weights()[0]

    weights1 = weights1.reshape(7, 7, 512, 4096)
    weights2 = weights2.reshape(1, 1, 4096, 4096)

    dense_convs.layers[0].set_weights([weights1])
    dense_convs.layers[2].set_weights([weights2])

    fcn32s_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    fcn32s_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Maintain mIOU and Pixel-wise Accuracy as metrics
    fcn32s_model.compile(
        optimizer=fcn32s_optimizer,
        loss=fcn32s_loss,
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ],
    )

    fcn32s_model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds)

    # Save the trained model
    fcn32s_model.save("fcn32s_model.h5")

def load_and_predict_model():
    # بارگذاری مدل آموزش داده شده
    fcn32s_model = load_model("fcn32s_model.h5")
    return fcn32s_model

# این بخش فقط یک بار برای آموزش و ذخیره مدل اجرا می‌شود.
if not os.path.exists("fcn32s_model.h5"):
    train_and_save_model()

# بارگذاری مدل برای پیش‌بینی
fcn32s_model = load_and_predict_model()

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.vgg19.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_mask(image, model):
    pred_mask = model.predict(image).astype("float")
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0, ...]
    return pred_mask

def mask_to_image(mask, colormap=plt.cm.viridis):
    normed_mask = mask / np.max(mask)  # نرمال‌سازی ماسک
    colormap_mask = colormap(normed_mask)  # اعمال colormap
    image_array = (colormap_mask[:, :, :3] * 255).astype(np.uint8)  # تبدیل به uint8 برای ساختن تصویر
    return Image.fromarray(image_array)

@router.post("/predictfcn/")
async def predict(file: UploadFile = File(...)):
    image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)

    mask_32s = predict_mask(preprocessed_image, fcn32s_model)

    # تبدیل ماسک به تصویر
    mask_image = mask_to_image(mask_32s)

    # ذخیره تصویر در حافظه به عنوان بایت‌ها
    img_byte_arr = io.BytesIO()
    mask_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")
