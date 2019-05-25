# -*- coding: utf-8 -*-
"""
Deep Mice Pose Estimation Using Stacked Hourglass Network

Project by @Eason
Adapted from @Walid Benbihi [source code]github : https://github.com/wbenbihi/hourglasstensorlfow/

---
Evaluation on a single image function
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
    
    cv2.imwrite(os.path.join(params['test_img_directory'], name), img)

if __name__ == '__main__':
    print('-- Parsing Config File')
    params = process_config('./config.cfg')
    model = Inference(model = params['pretrained_model'])

    img = cv2.imread(os.path.join(params['test_img_directory'], 'cam_00_0.2_30_001.png'))
    bbox = [540,1000,850,1310]
    img_crop = np.copy(img)[bbox[0]:bbox[1], bbox[2]:bbox[3]] # Assume: given bounding box
    test_img = cv2.resize(img_crop, (256, 256))
    #test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    predictions = model.predictJoints(test_img, mode='gpu')
    print(np.add(np.asarray(predictions)*(bbox[1]-bbox[0])/256, np.array([bbox[0], bbox[2]])))
    show_prections(test_img, predictions,'cam_00_0.2_30_001_r.png')