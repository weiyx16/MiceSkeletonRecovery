# -*- coding: utf-8 -*-
"""
Deep Mice Pose Estimation Using Stacked Hourglass Network

Project by @Eason
Adapted from @Walid Benbihi [source code]github : https://github.com/wbenbihi/hourglasstensorlfow/

---
Evaluation function
---
"""

# TODO: maybe you can add in yolo?

import sys
sys.path.append('./')

from hourglass_tiny import HourglassModel
from time import time, clock
import numpy as np
import tensorflow as tf
import scipy.io
from train_launcher import process_config
import cv2
from predictClass import PredictProcessor
from datagen import DataGenerator

class Inference():
	""" Inference Class
	Use this file to make your prediction
	Easy to Use
	Images used for inference should be RGB images (int values in [0,255])
	Methods:
		webcamSingle : Single Person Pose Estimation on Webcam Stream
		webcamMultiple : Multiple Person Pose Estimation on Webcam Stream
		webcamPCA : Single Person Pose Estimation with reconstruction error (PCA)
		# webcamYOLO : Object Detector
		predictHM : Returns Heat Map for an input RGB Image
		predictJoints : Returns joint's location (for a 256x256 image)
		pltSkeleton : Plot skeleton on image
	"""
	def __init__(self, config_file = 'config.cfg', model = 'mice_tiny_hourglass-49'):
		""" Initilize the Predictor
		Args:
			config_file 	 	: *.cfg file with model's parameters
			model 	 	 	 	: *.data-00000-of-00001 file's name. (weights to load) 
			# yoloModel 	 	: *.ckpt file (YOLO weights to load)
		"""
		t = time()
		params = process_config(config_file)
		model = params['pretrained_model']
		self.predict = PredictProcessor(params)
		self.predict.color_palette()
		self.predict.LINKS_JOINTS()
		self.predict.model_init()
		self.predict.load_model(load = model)
		self.predict._create_prediction_tensor()
		print('-- Prediction Intialization Done: ', time() - t, ' sec.')
		del t
		
	# -------------------------- WebCam Inference-------------------------------
	def webcamSingle(self, thresh = 0.2, pltJ = True, pltL = True):
		""" Run Single Pose Estimation on Webcam Stream
		Args :
			thresh: Joint Threshold
			pltJ: (bool) True to plot joints
			pltL: (bool) True to plot limbs
		"""
		self.predict.hpeWebcam(thresh = thresh, plt_j = pltJ, plt_l = pltL, plt_hm = False, debug = False)
	
	def webcamMultiple(self, thresh = 0.2, nms = 0.5, resolution = 800,pltL = True, pltJ = True, pltB = True, isolate = False):
		""" Run Multiple Pose Estimation on Webcam Stream
		Args:
			thresh: Joint Threshold
			nms : Non Maxima Suppression Threshold
			resolution : Stream Resolution
			pltJ: (bool) True to plot joints
			pltL: (bool) True to plot limbs
			pltB: (bool) True to plot bounding boxes
			isolate: (bool) True to show isolated skeletons
		"""
		self.predict.mpe(j_thresh = thresh, nms_thresh = nms, plt_l = pltL, plt_j = pltJ, plt_b = pltB, img_size = resolution, skeleton = isolate)
	
	def webcamPCA(self, n = 5, matrix = 'p4frames.mat'):
		""" Run Single Pose Estimation with Error Reconstruction on Webcam Stream
		Args:
			n : Number of dimension to keep before reconstruction
			matrix : MATLAB eigenvector matrix to load
		"""
		self.predict.reconstructACPVideo(load = matrix, n = n)
		
	# ----------------------- Heat Map Prediction ------------------------------
	
	def predictHM(self, img):
		""" Return Sigmoid Prediction Heat Map
		Args:
			img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
		"""
		return self.predict.pred((img.astype(np.float32) / 255), sess = None)
	# ------------------------- Joint Prediction -------------------------------
	
	def predictJoints(self, img, mode = 'cpu', thresh = 0.2):
		""" Return Joint Location
		! Location with respect to 256x256 image
		Args:
			img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
			mode : 'cpu' / 'gpu' Select a mode to compute joints' location
			thresh : Joint Threshold
		"""
		SIZE = False
		# fit to the model input (none*256*256*3)
		if len(img.shape) == 3:
			batch = np.expand_dims(img, axis = 0)
			SIZE = True
		elif len(img.shape) == 4:
			batch = np.copy(img)
			SIZE = True
		if SIZE:
			if mode == 'cpu':
				return self.predict.joints_pred_numpy(batch / 255, coord = 'img', thresh = thresh, sess = None)
			elif mode == 'gpu':
				return self.predict.joints_pred(batch / 255, coord = 'img', sess = None)
			else :
				print("Error : Mode should be 'cpu'/'gpu'")
		else:
			print('Error : Input is not a RGB image nor a batch of RGB images')
	
	# ----------------------------- Plot Skeleton ------------------------------
	
	def pltSkeleton(self, img, thresh, pltJ, pltL):
		""" Return an image with plotted joints and limbs
		Args:
			img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255]) 
			thresh: Joint Threshold
			pltJ: (bool) True to plot joints
			pltL: (bool) True to plot limbs
		"""
		return self.predict.pltSkeleton(img, thresh = thresh, pltJ = pltJ, pltL = pltL, tocopy = True, norm = True)
	
	# -------------------------- Video Processing ------------------------------
	
	def processVideo(self, source = None, outfile = None, thresh = 0.2, nms = 0.5 , codec = 'DIVX', pltJ = True, pltL = True, pltB = True, show = False):
		""" Run Multiple Pose Estimation on Video Footage
		Args:
			source : Input Footage
			outfile : File to Save
			thesh : Joints Threshold
			nms : Non Maxima Suppression Threshold
			codec : Codec to use
			pltJ: (bool) True to plot joints
			pltL: (bool) True to plot limbs
			pltB: (bool) True to plot bounding boxes
			show: (bool) Show footage during processing
		"""
		return self.predict.videoDetection(src = source, outName = outfile, codec = codec, j_thresh = thresh, nms_thresh = nms, show = show, plt_j = pltJ, plt_l = pltL, plt_b = pltB)
	
	# -------------------------- Process Stream --------------------------------
	
	def centerStream(self, img):
		img = cv2.flip(img, 1)
		img[:, self.predict.cam_res[1]//2 - self.predict.cam_res[0]//2:self.predict.cam_res[1]//2 + self.predict.cam_res[0]//2]
		img_hg = cv2.resize(img, (256,256))
		img_res = cv2.resize(img, (800,800))
		img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
		return img_res, img_hg
	
	def plotLimbs(self, img_res, j):
		""" 
		"""	
		for i in range(len(self.predict.links)):
			l = self.predict.links[i]['link']
			good_link = True
			for p in l:
				if np.array_equal(j[p], [-1,-1]):
					good_link = False
			if good_link:
				pos = self.predict.givePixel(l, j)
				cv2.line(img_res, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.predict.links[i]['color'][::-1], thickness = 5)
			
				
		
		
		
		