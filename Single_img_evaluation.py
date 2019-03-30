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

RED = (0, 0, 255)

def show_prections(img, predictions):
    for index, coord in enumerate(predictions):
        keypt = (int(coord[1]), int(coord[0]))
        text_loc = (keypt[0]+7, keypt[1]+7)
        cv2.circle(img, keypt, 3, RED, -1)
        cv2.putText(img, str(index), text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, RED, 1, cv2.LINE_AA)
    
    cv2.imwrite(os.path.join(params['text_img_directory'], 'test_result.png'), img)

if __name__ == '__main__':
    print('-- Parsing Config File')
    params = process_config('config.cfg')
    
    img = cv2.imread(os.path.join(params['text_img_directory'], 'test.png'))

    img_crop = np.copy(img)[480:988, 900:1408]
    test_img = cv2.resize(img_crop, (256, 256))
    #test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    model = Inference()

    predictions = model.predictJoints(test_img, mode='gpu')
    print(predictions)

    show_prections(test_img, predictions)