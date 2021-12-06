'''
BUI MAI QUYNH LINH
Pre trained: Mobilenetv2 as backbone
-- face detection
-- keep face and delete everything
-- trained
'''
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB1

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
import neptune.new as neptune
from mtcnn import MTCNN

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

def face_detection(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    return faces

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_dir = './train/'
    model_link = 'mobile_facemask_1'
    # train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, color_mode='rgb', labels='inferred',
    #                                                        label_mode='categorical',
    #                                                        subset='training', batch_size=32,
    #                                                        image_size=(96, 96), seed=1, validation_split=0.2)

    # val_ds = tf.keras.utils.image_dataset_from_directory(train_dir, color_mode='rgb', labels='inferred',
    #                                                      label_mode='categorical',
    #                                                      subset='validation', batch_size=32,
    #                                                      image_size=(96, 96), seed=1, validation_split=0.2)
    # rescale = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1)
    # train_ds = train_ds.map(lambda image, label: (rescale(image), label))

    model = model_define()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    model.load_weights('mobile_facemask_1.h5')
    train_meta = train_dir + 'train_meta_mask.csv'
    data = pd.read_csv(train_meta, sep=',')
    for idx in range(41):  # data.size):4175
        img = plt.imread(train_dir + 'images/' + data.iloc[
            idx, 2])  # , cv2.COLOR_BGR2RGB) # image seems like BGR. MTCNN trained with RGB
        print(data.iloc[idx, 2])
        faces = face_detection(img)
        i = 0
        for face in faces:
            i += 1
            face_box = face['box']
            h = np.round(face_box[2] * 0.2).astype(int)
            print(face['confidence'])
            if face['confidence'] > 0.9:
                try:
                    face_save = img[(face_box[1] - h): (face_box[1] + face_box[3] + 2 * h),
                                (face_box[0] - h):(face_box[0] + face_box[2] + 2 * h), :]
                    img_save = cv2.resize(face_save, (96, 96))
                    print(model.predict(face))
                    # if model.predict(face) == int(0.0):
                    #     plt.imsave(
                    #         train_dir + 'image_mask_224/0/' + data.iloc[idx, 2].replace('.jpg', '') + '_f_' + str(
                    #             i) + '.jpg', img_save)
                    # else:
                    #     plt.imsave(
                    #         train_dir + 'image_mask_224/1/' + data.iloc[idx, 2].replace('.jpg', '') + '_f_' + str(
                    #             i) + '.jpg', img_save)
                except:
                    continue

            else:
                continue
    # with tf.device('/GPU:0'):



if __name__ == '__main__':
    main()
