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
    input_1 = tf.keras.Input(shape=(96, 96, 3))
    input_2 = tf.keras.Input(shape=(96, 96, 3))

    base_model_1 = MobileNetV2(include_top=False, weights='imagenet', input_shape=(96, 96, 3), input_tensor=input_1)
    x_1 = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model_1.output)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)


    base_model_2 = MobileNetV2(include_top=False, weights='imagenet', input_shape=(96, 96, 3), input_tensor=input_2)
    x_2 = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model_2.output)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)

    concatted = tf.keras.layers.Concatenate()([x_1, x_2])
    top_dropout_rate = 0.2
    concatted = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(concatted)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred")(concatted)
    # Compile
    model = tf.keras.Model([input_1, input_2], outputs)
    return model

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_dir = './train/image_distance'
    model_link = 'mobilev2_both_2'
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, color_mode='rgb', labels='inferred',
                                                           label_mode='categorical',
                                                           subset='training', batch_size=32,
                                                           image_size=(96, 96), seed=1, validation_split=0.2)

    val_ds = tf.keras.utils.image_dataset_from_directory(train_dir, color_mode='rgb', labels='inferred',
                                                         label_mode='categorical',
                                                         subset='validation', batch_size=32,
                                                         image_size=(96, 96), seed=1, validation_split=0.2)
    rescale = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1)
    train_ds = train_ds.map(lambda image, label: (rescale(image), label))
    val_ds = val_ds.map(lambda image, label: (rescale(image), label))
    for x, y in train_ds.take(1):
        print('Image --> ', x.shape, 'Label --> ', y.shape)
    model = model_define()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    callbacks = [
        # EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, min_lr=0.00001, verbose=1),
        ModelCheckpoint(model_link + '.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    # with tf.device('/GPU:0'):
    history = model.fit(
        ([train_ds, train_dir], train_dir),
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks
    ).history
    pd.DataFrame.from_dict(history).to_csv('history' + model_link + '.csv', index=False)
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