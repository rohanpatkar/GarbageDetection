import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import glob
import argparse

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
#from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import time

def inference_image(img, image_name):
    PATH_TO_CKPT = '/media/sf_VMShared/garbage/frozen_inference_graph.pb'
    PATH_TO_LABELS = '/media/sf_VMShared/Road/Arrow/ObjectDetection/GarbageDetector/annotations/label_map.pbtxt'
    NUM_CLASSES = 1

    cap = cv2.VideoCapture(0)

    #PATH_TO_TEST_IMAGES_DIR = '/media/sf_srp/agc/ws/GarbageDetector/cam_images'
    PATH_TO_RESULT_IMAGES_DIR = r'/media/sf_VMShared/Road/Arrow/ObjectDetection/GarbageDetector/amar_garbage_road_inference/'

    from os.path import exists

    if not exists(PATH_TO_RESULT_IMAGES_DIR):
        os.makedirs(PATH_TO_RESULT_IMAGES_DIR)


    #img_np = load_image_into_numpy_array(img)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    vidcap = cv2.VideoCapture(r'/media/sf_VMShared/garbage_video/IMG_1444.MOV')
    count = 0
    success = True

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while success:
                #vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))
                success, img_np = vidcap.read()

                ## Stop when last frame is identified
                image_name = "frame{}.png".format(count - 1)
                count += 1
                if count == 610:
                    break;
                img_np = cv2.resize(img_np, (512,384))

                # Definite input and output Tensors for detection_graph
                #ret, img_np = cap.read()
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.

                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(img_np, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                print(scores)
                from os.path import join

                vis_util.visualize_boxes_and_labels_on_image_array(
                    img_np, join(PATH_TO_RESULT_IMAGES_DIR, image_name), img_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                cv2.imwrite(join(PATH_TO_RESULT_IMAGES_DIR, image_name), img_np)
                #cv2.imshow('object detection',img_np)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    print(str(count))


def load_image_into_numpy_array(image):
       return cv2.resize(image, (384,512))


def capture_frames_and_predict():
    sleep_interval = 10
    cam = cv2.VideoCapture(0)
    i = 0
    while True:
        i = i + 1
        _, img = cam.read()
        #img = cv2.imread(r'/media/sf_VMShared/Selection_001.png')#cam.read()
        img = cv2.flip(img, 1)
        inference_image(img, str(100) + '.jpg')
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        time.sleep(sleep_interval)


def predict_on_static_images(run_dir, out_dir = None) :
    from os import listdir
    images = listdir(run_dir)
    for image_name in images:
        image = cv2.imread(os.path.join(run_dir, image_name))  # cam.read()
        inference_image(image, image_name)

if __name__ == '__main__':
    #capture_frames_and_predict()
    #predict_on_static_images(r'/home/ubuntu/Rohan/Lane/GarbageDetector/images/')
    inference_image(None, None)
