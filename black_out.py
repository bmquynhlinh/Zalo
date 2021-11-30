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

def black_out(img_inv, face_box):
    mask = np.zeros_like(img_inv)
    h = np.round(face_box[2] * 0.2).astype(int)
    print(face_box[0] - h)
    mask = cv2.rectangle(mask, (face_box[0] - h, face_box[1] - h),
                         (face_box[0] + face_box[2] + 2 * h, face_box[1] + face_box[3] + 2 * h),
                         (255, 255, 255), -1)
    result = cv2.bitwise_or(img_inv, mask)
    return result


def get_img(img):
    return None


def inv_combine(img_main, img_mask):
    # img_mask[np.where((img_mask!=[0, 0, 0]).all(axis=2))] = [255,255,255]
    # cv2.imshow('name', img_mask)
    # cv2.waitKey(500)
    result = cv2.bitwise_and(img_main, img_mask)
    return result


def save_new():
    return


def main():
    train_dir = './train/'
    train_meta = train_dir + 'train_meta_mask.csv'
    data = pd.read_csv(train_meta, sep=',')
    if os.path.isfile('./train/train_mask_enet.csv'):
        with open('./train/train_mask_enet224.csv', 'w', newline='') as file:
            write = csv.writer(file)
            write.writerow(['frame_id', 'image_id', 'fname','f_image_name', 'mask'])
    with open('./train/train_mask_enet224.csv', 'a', newline='') as file:
        write = csv.writer(file)
        for idx in range(4175):  # data.size):4175
            print(data.iloc[idx, 2])
            try:
                img = plt.imread(train_dir + 'images/' + data.iloc[
                    idx, 2])  # , cv2.COLOR_BGR2RGB) # image seems like BGR. MTCNN trained with RGB
            except:
                continue
            faces = face_detection(img)
            img_inv = np.zeros_like(img)
            i = 0
            for face in faces:
                i += 1
                print(i)
                face_box = face['box']
                h = np.round(face_box[2] * 0.2).astype(int)
                print(face_box[0] - h)
                try:
                    face_save = img[ (face_box[1] - h): (face_box[1] + face_box[3] + 2 * h), (face_box[0] - h):(face_box[0] + face_box[2] + 2 * h), :]
                    img_save = cv2.resize(face_save, (224, 224))
                    if int(data.iloc[idx, 3])== int(0.0):
                        plt.imsave(train_dir + 'images_mask_224/0/' + data.iloc[idx, 2].replace('.jpg','') +'_f_'+ str(i)+'.jpg', img_save)
                        write.writerow([str(i), data.iloc[idx, 1], data.iloc[idx, 2], '0/'+data.iloc[idx, 2].replace('.jpg','') +'_f_'+ str(i)+'.jpg', data.iloc[idx, 3]])
                    else:
                        plt.imsave(train_dir + 'images_mask_224/1/' + data.iloc[idx, 2].replace('.jpg', '') + '_f_' + str(
                            i) + '.jpg', img_save)
                        write.writerow([str(i), data.iloc[idx, 1], data.iloc[idx, 2],'1/'+
                                        data.iloc[idx, 2].replace('.jpg', '') + '_f_' + str(i) + '.jpg',
                                        data.iloc[idx, 3]])
                except:
                    continue
        #     print(face['box'])
        #     img_inv = black_out(img_inv, face['box'])
        # img_save = inv_combine(img, img_inv)
        # img_save = cv2.resize(img_save, (32, 32))
        # plt.imsave(train_dir + 'images_inv/' + data.iloc[idx, 1], img_save)


if __name__ == '__main__':
    main()
