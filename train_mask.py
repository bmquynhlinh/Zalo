'''
BUI MAI QUYNH LINH
Pre trained: Mobilenetv2 as backbone
-- face detection
-- keep face and delete everything
-- trained
'''

from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB3
import tensorflow as tf


def model_define():
    inputs = (300, 300, 3)
    base_model = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred")(x)
    # Compile
    model = tf.keras.Model(inputs, outputs)
    return model

def loading(train_link):
    

    return None

def training():

    return None

def main():
    train_link = './train/train_meta_mask.csv'


if __name__ =='__main__':
    main()