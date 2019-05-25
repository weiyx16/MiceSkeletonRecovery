# -*- coding: utf-8 -*-
"""
Deep Mice Pose Estimation Using Stacked Hourglass Network

Project by @Eason
Adapted from @Walid Benbihi [source code]github : https://github.com/wbenbihi/hourglasstensorlfow/

---
Evaluation on set pf image function
---
"""

import os
from inference import Inference
from train_launcher import process_config
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
import numpy as np
import tensorflow as tf
import cv2

Palette = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]

def show_prections(img, predictions, name):
    for index, coord in enumerate(predictions):
        keypt = (int(coord[1]), int(coord[0]))
        text_loc = (keypt[0]+7, keypt[1]+7)
        cv2.circle(img, keypt, 3, Palette[index], -1)
        cv2.putText(img, str(index), text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, Palette[index], 1, cv2.LINE_AA)
    
    cv2.imwrite(os.path.join(params['test_result_directory'] , name), img)

if __name__ == '__main__':
    print('-- Parsing Config File')
    params = process_config('./config.cfg')
    model = Inference(model = params['pretrained_model'])
    input_file = open(params['training_txt_file'], 'r')
    bounding_box = {}
    print('-- Read training data and Convert it to a table')
    for line in input_file:
        line = line.strip().split(' ')
        bbox = list(map(int, line[1:5]))
        img_name = line[0]
        img_name = img_name[0:-1]
        bounding_box.update({img_name: bbox})
    img_paths = os.listdir(params['test_img_directory'])
    for img_path in img_paths:
        img = cv2.imread(os.path.join(params['test_img_directory'], img_path))
        bbox = bounding_box[img_path]
        size = max((bbox[2] - bbox[0]), (bbox[3] - bbox[1]))
        img_crop = np.copy(img)[bbox[1]:bbox[1]+size, bbox[0]:bbox[0]+size]
        img_resize = cv2.resize(img_crop, (256, 256))
        #test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        predictions = model.predictJoints(img_resize, mode='gpu')
        print(' Predict on {}' .format(img_path))
        print(np.add(np.asarray(predictions)*(bbox[1]-bbox[0])/256, np.array([bbox[0], bbox[2]])))
        show_prections(img_resize, predictions, img_path[0:-4] + '_pred' + img_path[-4:])