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
import matplotlib
import numpy as np
from mtcnn.mtcnn import MTCNN
import pandas as pd
from matplotlib import pyplot as plt, pyplot
# from face_toolbox_keras.models.detector import face_detector
import csv
from Yolo_1 import make_yolov3_model
from tensorflow.keras.models import load_model


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
# Detect human, return black image with red border and green filled

def load_image(filename):
    # load the image to get its shape
    img = plt.imread(filename)
    img = cv2.resize(img, (416, 416))
    img = img.astype('float32')
    img /= 255.0
    # add a dimension so that we have one sample
    img = np.expand_dims(img, 0)
    return img


def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = matplotlib.patches.Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def human_detection(link):
    model = load_model('model.h5')
    image = load_image(link)
    yhat = model.predict(image)
    class_threshold = 0.9
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    for i in range(len(yhat)):
        netout = yhat[i][0]
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2] = _sigmoid(netout[..., :2])
        netout[..., 4:] = _sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > 0.9
        print(netout[int(row)][col][b][5:] )
        # if yhat[i][0] > 0.9:
        #     print(yhat)
    # draw_boxes(link, v_boxes, v_labels, v_scores)
    return None


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    train_dir = './train/'
    train_meta = train_dir + 'train_meta_mask.csv'
    data = pd.read_csv(train_meta, sep=',')

    with open('./train/train_meta_mask.csv', 'a', newline='') as file:
        for idx in range(4174):  # data.size):4175
            link = train_dir + 'images/' + data.iloc[idx, 2]
            # human_detection(link)
            img_save = plt.imread(link)
            # img = cv2.resize(img, (416, 416))
            if int(data.iloc[idx, 3]) == int(0.0):
                plt.imsave(train_dir + 'images_mask/0/' + data.iloc[idx, 2], img_save)
            else:
                plt.imsave(train_dir + 'images_mask/1/' + data.iloc[idx, 2], img_save)


if __name__ == '__main__':
    main()
