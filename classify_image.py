'''
BUI MAI QUYNH LINH
Pre trained: Mobilenetv2 as backbone
-- face detection
-- keep face and delete everything
-- trained

human detection with YOLO
Save centroid as blue and black image

'''
import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import pandas as pd
from matplotlib import pyplot as plt
# from face_toolbox_keras.models.detector import face_detector
import csv


def face_detection(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    return faces


# def face_net(img):
#     fd = face_detector.FaceAlignmentDetector(
#         lmd_weights_path="./face_toolbox_keras/models/detector/s3fd/s3fd_keras_weights.h5"
#         # 2DFAN-4_keras.h5, 2DFAN-1_keras.h5
#     )
#     faces = fd.detect_face(img, with_landmarks=False)
#     return faces


# def haar(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     detector = cv2.CascadeClassifier('D:\Personal Plan\Program\Zalo\paper\haarcascade_frontalface_default.xml')
#     faces = detector.detectMultiScale(gray, 1.3, 5)
#     return faces
def human_detection():
    
    return None

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    train_dir = './train/'
    train_meta = train_dir + 'train_meta_distancing.csv'
    data = pd.read_csv(train_meta, sep=',')

    with open('./train/train_meta_distancing.csv', 'a', newline='') as file:
        for idx in range(2903):  # data.size):4175
            img = plt.imread(train_dir + 'images/' + data.iloc[idx, 2])  # , cv2.COLOR_BGR2RGB) # image seems like BGR. MTCNN trained with RGB
            face_save = img
            img_save = cv2.resize(face_save, (224, 224))
            if int(data.iloc[idx, 3]) == int(0.0):
                plt.imsave(train_dir + 'image_distance/0/' + data.iloc[idx, 2], img_save)
            else:
                plt.imsave(train_dir + 'image_distance/1/' + data.iloc[idx, 2], img_save)


if __name__ == '__main__':
    main()
