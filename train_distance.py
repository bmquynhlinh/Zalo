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
from tensorflow.keras.applications.efficientnet import EfficientNetB3
import tensorflow as tf
import pandas as pd
import matplotlib as plt
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
import neptune.new as neptune

def model_define():
    inputs = tf.keras.Input(shape=(96, 96, 3))
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(96, 96, 3), input_tensor=inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred")(x)
    # Compile
    model = tf.keras.Model(inputs, outputs)
    return model


def YOLO_human():
    return None

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_dir = './train/image_distance'
    model_link = 'mobilev2_distance_2'
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, color_mode='rgb', labels='inferred',label_mode ='categorical',
                                                           subset='training', batch_size=32,
                                                           image_size=(96, 96), seed=1, validation_split=0.2)

    val_ds = tf.keras.utils.image_dataset_from_directory(train_dir, color_mode='rgb', labels='inferred',label_mode ='categorical',
                                                         subset='validation', batch_size=32,
                                                         image_size=(96, 96), seed=1, validation_split=0.2)
    for x, y in train_ds.take(1):
        print('Image --> ', x.shape, 'Label --> ', y.shape)
    model = model_define()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, min_lr=0.00001, verbose=1),
        ModelCheckpoint( model_link + '.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    # with tf.device('/GPU:0'):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks
    ).history
    pd.DataFrame.from_dict(history).to_csv('history'+model_link+'.csv', index=False)
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
