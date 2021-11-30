'''
BUI MAI QUYNH LINH
Pre trained: Mobilenetv2 as backbone
-- face detection
-- keep face and delete everything
-- trained
'''
import os

from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
import pandas as pd
import matplotlib as plt
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
import neptune.new as neptune

def model_define():
    inputs = tf.keras.Input(shape=(96, 96, 3))
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(96,96,3), input_tensor=inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred")(x)
    # Compile
    model = tf.keras.Model(inputs, outputs)
    return model




def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    train_dir = './train/images_mask'
    model_link = 'mobilenetv2_facemask3'
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, color_mode='rgb',
                                                           subset='training', batch_size=32,
                                                           image_size=(96, 96), seed=123, validation_split=0.2)

    val_ds = tf.keras.utils.image_dataset_from_directory(train_dir, color_mode='rgb',
                                                         subset='validation', batch_size=32,
                                                         image_size=(96, 96), seed=123, validation_split=0.2)
    model = model_define()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=['accuracy'])
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, min_lr=0.00001, verbose=1),
        ModelCheckpoint( model_link + '.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    with tf.device('/CPU:0'):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=callbacks
        )
    run = neptune.init(
        project="bmquynhlinh/zalo",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1OTNjYmFhZC0yZmJkLTRkYTQtYjdlYi1jODRkNzAwMzcxMWIifQ==",
    )  # your credentials
    for epoch in range(len(history['accuracy'])):
        run['train/epoch accuracy'].log(history['accuracy'][epoch])
        run['train/epoch loss'].log(history['loss'][epoch])
        run['train/epoch val_accuracy'].log(history['val_accuracy'][epoch])
        run['train/epoch val_loss'].log(history['val_loss'][epoch])

if __name__ == '__main__':
    main()
