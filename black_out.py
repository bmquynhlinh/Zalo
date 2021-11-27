'''
BUI MAI QUYNH LINH
Pre trained: Mobilenetv2 as backbone
-- face detection
-- keep face and delete everything
-- trained
'''
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import pandas as pd
from PIL import Image


def face_detection(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    return faces


def black_out(img_inv, face_box):
    mask = np.zeros_like(img_inv)
    mask = cv2.rectangle(mask, (face_box[0], face_box[1]),(face_box[0]+face_box[2], face_box[1]+face_box[3]), (255,255,255),-1)
    result = cv2.bitwise_xor(img_inv, mask)
    return result


def inv_combine(img_main, img_mask):
    # img_mask[np.where((img_mask!=[0, 0, 0]).all(axis=2))] = [255,255,255]
    cv2.imshow('name', img_mask)
    cv2.waitKey(500)
    result = cv2.bitwise_and(img_main, img_mask)
    return result


def main():
    train_dir = './train/'
    train_meta = train_dir + 'train_meta.csv'
    data = pd.read_csv(train_meta, sep=',')
    for idx in range(2): #data.size):
        print(data.iloc[idx, 1])
        try:
            img = Image.open(train_dir + 'images/' + data.iloc[idx, 1])#, cv2.COLOR_BGR2RGB) # image seems like BGR
        except:
            continue
        faces = face_detection(img)
        img_inv = np.zeros_like(img)
        for face in faces:
            print(face['box'])
            img_inv = black_out(img_inv, face['box'])
        img_save = inv_combine(img, img_inv)
        cv2.imwrite(train_dir + 'images_inv/' + data.iloc[idx, 1], img_save)


if __name__ == '__main__':
    main()
