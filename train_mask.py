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

def loading(train_link):
    train_meta = './train/train_mask_enet.csv'
    data = pd.read_csv(train_meta, sep=',', header= None)

    for idx in range(data.shape[0]):
        x =  plt.imread('./train/images_mask/' + data.iloc[idx, 3])
        y = data.iloc[idx, 4]



    return None

def training():

    return None

def main():
    train_link = './train/train_meta_mask.csv'


if __name__ =='__main__':
    main()