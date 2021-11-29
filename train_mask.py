'''
BUI MAI QUYNH LINH
Pre trained: Mobilenetv2 as backbone
-- face detection
-- keep face and delete everything
-- trained
'''

from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
import pandas as pd
import matplotlib as plt


def model_define():
    inputs = (64, 64, 3)
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred")(x)
    # Compile
    model = tf.keras.Model(inputs, outputs)
    return model




def main():
    train_dir = './train/images_mask'
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, label_mode='int', color_mode='rgb',
                                                           subset='training', batch_size='32',
                                                           image_size=(64, 64), seed=123, validation_split=0.2)

    val_ds = tf.keras.utils.image_dataset_from_directory(train_dir, label_mode='int', color_mode='rgb',
                                                         subset='validation', batch_size='32',
                                                         image_size=(64, 64), seed=123, validation_split=0.2)
    model = model_define()
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    with tf.device('/CPU:0'):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30
        )


if __name__ == '__main__':
    main()
