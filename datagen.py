# -*- coding: utf-8 -*-
"""
Deep Mice Pose Estimation Using Stacked Hourglass Network

Project by @Eason
Adapted from @Walid Benbihi [source code]github : https://github.com/wbenbihi/hourglasstensorlfow/

---
Dataset Creation function
---
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math
import random
import time
# from skimage import transform
import scipy.misc as scm
import json
import yaml
import scipy

class DataGenerator():
	""" 
	DataGenerator Class: To generate Train, Validatidation and Test sets for the Deep Mice Pose Estimation Model 
	
	The Generator will read the TEXT file(dataset) to create a dictionary
		Inputs:
			dataset_path: train_data_file
			img_directory: img_dir
			joints_lists: joints_name
	Then 2 options are available for training:
		Store image/heatmap arrays (numpy file stored in a folder: need disk space but faster reading)
		Generate image/heatmap arrays when needed (Generate arrays while training, increase training time - Need to compute arrays at every iteration) 
	
	Formalized DATA:
		Inputs:
			Shape of (Number of Image) X (Height: 256) X (Width: 256) X (Channels: 3(RGB))
		Outputs:
			Shape of (Number of Image) X (Number of Stacks) X (Heigth: 64) X (Width: 64) X (Output Dimension: 9(joints))
	
	Joints definition:
		00 - 'nose'
		01 - 'r_ear'
		02 - 'l_ear'
		03 - 'rf_leg'
		04 - 'lf_leg'
		05 - 'rb_leg'
		06 - 'lb_leg'
		07 - 'tail_base'
		08 - 'tail_end'
	
	How to generate Dataset:
		Create a TEXT file with the following structure:
			image_name.jpg[LETTER] box_xmin box_ymin box_xmax b_ymax joints
			[LETTER]: (Not need yet)
				One image can contain multiple mice. To use the same image
				finish the image with a CAPITAL letter [A,B,C...] for 
				first/second/third... mice in the image
 			joints : 
				Sequence of x_p y_p (p being the p-joint)
				Missing values with -1
	"""

	def __init__(self, joints_name = None, img_dir=None, train_data_file = None, camera_extrinsic = None, camera_intrinsic = None):
		""" `Initializer`
		Args:
			joints_name			: List of joints condsidered
			img_dir				: Directory containing every images
			train_data_file		: Text file with training set data
			camera_extrinsic    : file with multicamera extrinsic data
			camera_intrinsic    : file with multicamera intrinsic data
		"""
		if joints_name == None:
			self.joints_list = ['nose','r_ear','l_ear','tail_base'] #['nose','r_ear','l_ear','rf_leg','lf_leg','rb_leg','lb_leg','tail_base','tail_end']
		else:
			self.joints_list = joints_name
		
		self.letter = ['A','B','C']
		self.img_dir = img_dir
		self.train_data_file = train_data_file
		self.images = os.listdir(img_dir)
		self.camera_extrinsic = camera_extrinsic
		self.camera_intrinsic = camera_intrinsic

	"""
	# --------------------Generator Initialization Methods ---------------------
	"""
	def _create_train_table(self):
		""" 
		Create Table of samples from TEXT file

		Args:
			train_table: save the names of images trainable (sum of train_set and validation_set)
			no_intel: save the names of images untrainable (all joint unvisible)
			data_dict: training data dictionary -- name: (box; joint(9*2); visibility(0/1))
		
		Notice:
			Only use the data with at least one visible joint and set the unvisible joint
			to weight = 0 which we can avoid calculating the loss during training
		"""
		self.train_table = [] # just save the names of images trainable
		self.no_intel = [] # save the images with no mice appears
		self.data_dict = {}
		self.img_pair = []
		input_file = open(self.train_data_file, 'r')
		dataset_to_list = []
		print('-- Read training data and Convert it to a table')
		for line in input_file:
			line = line.strip().split(' ')
			dataset_to_list.append(line)
		for line in dataset_to_list:
			if line[0][0:6] == r'cam_00':
				name_to_find = []
				for i in range(4):
					# 4: camera number
					name_to_find.append(line[0][0:5] + str(i) + line[0][6:])
				index_to_find = []
				for index, line_to_find in enumerate(dataset_to_list):
					for i in range(4):
						if name_to_find[i] == line_to_find[0]:
							index_to_find.append(index)
				cur_img_pair = name_to_find
				for i in range(4):
					line_to_find = dataset_to_list[index_to_find[i]]
					name = line_to_find[0]
					box = list(map(int,line_to_find[1:5]))
					joints = list(map(int,line_to_find[5:])) # convert each joint location to int
					# Fetch only 4 joints of 9 (Only Train on 4 Joints, 2019.04.19)
					joints = joints[:6]+joints[14:16]
					# reshape to 4 row 2 col 
					joints = np.reshape(joints, (-1,2))
					w = [1] * joints.shape[0]
					# set weights = 0 if this joint is not visiable
					for wi in range(joints.shape[0]):
						if np.array_equal(joints[wi], [-1,-1]):
							w[wi] = 0
					# in the form of dict with name+(box+joint+visibility)
					self.data_dict[name] = {'box' : box, 'joints' : joints, 'weights' : w}
					self.train_table.append(name)
				self.img_pair.append(cur_img_pair) # Use the cam_00 as the index of the pair in the dict
		input_file.close()
		self._camera_rectification_homography_params()

		# self._randomize()
		# self._create_sets()

		# Use it if you don't need validation set
		self.train_set = self.train_table
		print('-- Dataset Created')
		np.save('Dataset-Training-Set', self.train_set)
		print('-- Training set: ', len(self.train_set), ' samples.')
		print('-- Training image pairs: ', len(self.img_pair))


	def _randomize(self):
		""" 
		Randomize the trainset_table
		"""
		random.shuffle(self.train_table)
	
	def _complete_sample(self, joints):
		""" Check if a sample has all joints value, which means every joints are visible
		Args:
			joints 	: the sample joints
		"""
		for i in range(joints.shape[0]):
			if np.array_equal(joints[i,:],[-1,-1]):
				return False
		return True
	
	def _create_sets(self, validation_rate = 0.1):
		""" Select Elements to feed `training and validation set`
			Args:
			validation_rate	: Percentage of validation data in (0,1)
			Notice validation_set only consists of samples with all the joints
		# if the sample is not completed then make it as a training sample
		"""
		pass
		'''
		# The validation part have not been adapted to multiview version
		sample = len(self.train_table)
		valid_sample = int(sample * validation_rate)
		self.train_set = self.train_table[:sample - valid_sample]
		self.valid_set = []
		preset = self.train_table[sample - valid_sample:]
		print('-- Start Set Creation')
		for elem in preset:
			if self._complete_sample(elem):
				self.valid_set.append(elem)
			else:
				self.train_set.append(elem)
		print('-- Dataset Created')
		np.save('Dataset-Validation-Set', self.valid_set)
		np.save('Dataset-Training-Set', self.train_set)
		print('-- Training set :', len(self.train_set), ' samples.')
		print('-- Validation set :', len(self.valid_set), ' samples.')
		'''
	def _camera_rectification_homography_params(self):
		"""
			Load camera params and calculate the rectification homography matrix between any two-camera pair
			It is necessary to calculate the multiview reprojection loss insighted from epipolar constrains
			ref: Monet: multiview semi-supervised keypoint via epipolar divergence
		"""
		# Load data
		R = []
		T = []
		self.Extrinsic = []
		self.Intrinsic = []
		self.Rectification_Homography_Matrix = []
		# 4 is camera numbers
		with open(self.camera_extrinsic, encoding='utf-8') as f:
			extrinsic_file = json.load(f)
			for i in range(4):
				R_matrix,_ = cv2.Rodrigues(np.asarray(extrinsic_file['extrinsic_0'+str(i)]['R']))
				R.append(np.reshape(R_matrix, (3,3)))
				T.append(np.reshape(np.asarray(extrinsic_file['extrinsic_0'+str(i)]['T']), (3,1)))
				self.Extrinsic.append(np.concatenate((R[-1], T[-1]), axis = 1))

		with open(self.camera_intrinsic, encoding='utf-8') as f:
			intrinsic_file = yaml.load(f)
			for i in range(4):
				self.Intrinsic.append(np.reshape(np.asarray(intrinsic_file['K'+str(i)]['data']), (3,3)))

		# 只计算相邻视角的Rrect矩阵，相邻视角为:(00,02),(00,01),(01,03),(02,03)
		camera_pair = [[0,1],[0,2],[1,3],[2,3]]
		for pair_index in range(4):
			# (i,j)是旋转第i个视角到世界坐标系下，然后校正i与j视角的极线
			# 注意使用时，i与j都会先各自单独旋转到世界坐标系，然后用光心之间的平移去校正
			# 所以Hr矩阵是定义在两个（一对）相机之间的
			Homography_matrix_lr = [] # from i to other cameras j
			camera_l = camera_pair[pair_index][0]
			camera_r = camera_pair[pair_index][1]
			
			# Step 1: solve E = [t]_x * R
			# !!! Notice the negative and positive annotation (same with CMU tutorial 9)
			# 解法：Bouguet 参考opencv.cvStereoRectify,非标准解法
			# https://github.com/opencv/opencv/blob/19cf5118957cf9cb86022e44998f652ddaa5d887/modules/calib3d/src/calibration.cpp
			
			R_rel = np.dot(R[camera_r], np.transpose(R[camera_l]))
			R_l = scipy.linalg.sqrtm(R_rel) #R_r.T = R_l
			R_l = R_l.real
			R_r = np.linalg.inv(R_l)
			# T_rel = T[camera_l] - T[camera_r]
			# T_x = [0, -T_rel[2], T_rel[1],
			# 		T_rel[2], 0, -T_rel[0],
			# 		-T_rel[1], T_rel[0], 0]
			# T_x = np.reshape(list(map(float, T_x)), (3,3))				
			# E = np.dot(T_x, R_rel)
			# # Step 2: solve Epipole Ee = 0
			# _, S, V = np.linalg.svd(E)
			# e = np.compress(S == np.min(S), V, axis=0)

			t = np.dot(R_r, T[camera_r] - T[camera_l])
			if (abs(t[0])>abs(t[1])):
				idx = 0
			else:
				idx = 1

			e = t.T
			# Step 3: Rrect
			r_1_T = e / np.linalg.norm(e)
			r_2_T = np.asarray([[-e[0][1], e[0][0], 0]]) / math.sqrt(math.pow(e[0][1],2)+math.pow(e[0][0],2))
			r_3_T = np.cross(r_1_T, r_2_T, axis=1)
			if idx == 0:
				r_1_T = -r_1_T
				r_2_T = -r_2_T
				Rrect = np.concatenate((r_1_T, r_2_T, r_3_T), axis = 0)
			else:
				Rrect = np.concatenate((-r_2_T, -r_1_T, -r_3_T), axis = 0)
		
			Hr_l = np.dot(self.Intrinsic[camera_l], np.dot(np.dot(Rrect, R_r.T), np.linalg.inv(self.Intrinsic[camera_l])))
			Hr_r = np.dot(self.Intrinsic[camera_r], np.dot(np.dot(Rrect, R_r), np.linalg.inv(self.Intrinsic[camera_r])))

			Homography_matrix_lr.append(Hr_l)
			Homography_matrix_lr.append(Hr_r)
			self.Rectification_Homography_Matrix.append(Homography_matrix_lr)
		print(self.Rectification_Homography_Matrix)
	"""
	# ------------- Ground Truth HeatMap for each joints Creator ------------ 
	"""
	def _makeGaussian(self, height, width, sigma = 3.25, center=None):
		""" Make a square gaussian kernel.
		size is the length of a side of the square
		sigma is full-width-half-maximum, which
		can be thought of as an effective radius.
		"""
		x = np.arange(0, width, 1, float)
		y = np.arange(0, height, 1, float)[:, np.newaxis]
		if center is None:
			x0 =  width // 2
			y0 = height // 2
		else:
			x0 = center[0]
			y0 = center[1]
		return np.exp(-4*math.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)
	
	def _generate_hm(self, height, width ,joints, maxlenght, weight):
		""" Generate a full Heap Map for every joints in an array
		Args:
			height			: Wanted Height for the Heat Map
			width			: Wanted Width for the Heat Map
			joints			: Array of Joints
			maxlenght		: Lenght of the Bounding Box (set to 64)
			weight          : Joints visibility
		"""
		num_joints = joints.shape[0]
		hm = np.zeros((height, width, num_joints), dtype = np.float)
		for i in range(num_joints):
			if not(np.array_equal(joints[i], [-1,-1])) and weight[i] == 1:
				s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2 # if maxlenght is 64 then s = 3.25
				hm[:,:,i] = self._makeGaussian(height, width, sigma = s, center= (joints[i,0], joints[i,1]))
			else:
				hm[:,:,i] = np.zeros((height,width))
		return hm
	
	"""
	# ------------- IMG & Data Crop ------------------- 
	"""

	def open_img(self, name, color = 'RGB'):
		""" Load an image through opencv3 
		Args:
			name	: Name of the sample
			color	: Color Mode (RGB/BGR/GRAY)
		"""
		if name[-1] in self.letter:
			name = name[:-1]
		img = cv2.imread(os.path.join(self.img_dir, name))
		if color == 'RGB':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img
		elif color == 'BGR':
			return img
		elif color == 'GRAY':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		else:
			print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

	def _crop_data(self, height, width, box, joints, boxp = 0.1):
		""" Automatically returns a padding vector and a bounding box (adapted from labeled data and in the form of [x_center,y_center,x_length,y_length])
		Args:
			height		: Original Height
			width		: Original Width
			box			: Original bounding Box [x_min,y_min,x_max,y_max]
			joints		: Array of joints
			boxp		: Box percentage (Use 10% to get a good bounding box)
		"""
		padding = [[0,0],[0,0],[0,0]]
		j = np.copy(joints)
		if box[0:2] == [-1,-1]:
			j[joints == -1] = 1e5 # set un visible joint to biggest value without disturbance to create new bounding box
			box[0], box[1] = min(j[:,0]), min(j[:,1])

		# use boxp value to create a bounding box bigger than labeled one for safety.
		pad_x = int(boxp * (box[2]-box[0]))
		pad_y = int(boxp * (box[3]-box[1]))
		crop_box = [box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y]
		if crop_box[0] < 0: crop_box[0] = 0
		if crop_box[1] < 0: crop_box[1] = 0
		if crop_box[2] > width -1: crop_box[2] = width -1
		if crop_box[3] > height -1: crop_box[3] = height -1
		
		new_h = int(crop_box[3] - crop_box[1])
		new_w = int(crop_box[2] - crop_box[0])
		# convert from [x_min,y_min,x_max,y_max] to [x_center,y_center,x_length,y_length]
		crop_box = [crop_box[0] + new_w //2, crop_box[1] + new_h //2, new_w, new_h]
		if new_h > new_w:
			bounds = (crop_box[0] - new_h //2, crop_box[0] + new_h //2)
			if bounds[0] < 0:
				padding[1][0] = abs(bounds[0])
			if bounds[1] > width - 1:
				padding[1][1] = abs(width - bounds[1])
		elif new_h < new_w:
			bounds = (crop_box[1] - new_w //2, crop_box[1] + new_w //2)
			if bounds[0] < 0:
				padding[0][0] = abs(bounds[0])
			if bounds[1] > width - 1:
				padding[0][1] = abs(height - bounds[1])
		crop_box[0] += padding[1][0]
		crop_box[1] += padding[0][0]
		return padding, crop_box # padding is for img pad during crop (just in case)
	
	def _crop_img(self, img, padding, box):
		""" Given a bounding box and padding values return cropped image
		Args:
			img			: Source Image
			padding	    : Padding
			box	: Bounding Box
		"""
		img = np.pad(img, padding, mode = 'constant')
		max_l = max(box[2], box[3]) # choose max in width and height for sure that it's a square.
		crop_size = max_l //2
		img = img[box[1]-crop_size : box[1]+crop_size, box[0]-crop_size : box[0]+crop_size]
		return img
	
	def _relative_joints(self, box, padding, joints, to_size = 64):
		""" Convert Absolute joint coordinates to cropbox-related joint coordinates
		(Used to compute Heat Maps)
		Args:
			box			: Bounding Box 
			padding	: Padding Added to the original Image
			to_size	: Heat Map wanted Size
		"""
		new_j = np.copy(joints)
		max_l = max(box[2], box[3])
		new_j = new_j + [padding[1][0], padding[0][0]]
		new_j = new_j - [box[0] - max_l //2, box[1] - max_l //2]
		new_j = new_j * to_size / (max_l + 0.0000001)
		return new_j.astype(np.int32)
	
	"""
	# ----------------------- Batch Data Generator ----------------------------------
	"""

	def _aux_generator(self, batch_size = 4, stacks = 4, normalize = True, sample_set = 'train'):
		""" Auxiliary Generator `Create a method to get batch samples during training`
		Args:
				batch_size	: Number of images per batch
				stacks			: Number of stacks/module in the network
				normalize		: True to return Image Value between 0 and 1
		"""
		while True:
			train_img = np.zeros((4, 256, 256, 3), dtype = np.float)
			train_gtmap = np.zeros((4, stacks, 64, 64, len(self.joints_list)), np.float)
			train_weights = np.zeros((4, len(self.joints_list)), np.float) # visibility of each joint
			train_bbox = np.zeros((4, 4), dtype = np.float)
			train_img_pair = np.zeros((batch_size, 4, 256, 256, 3), dtype = np.float)
			train_gtmap_pair = np.zeros((batch_size, 4, stacks, 64, 64, len(self.joints_list)), np.float)
			train_weights_pair = np.zeros((batch_size, 4, len(self.joints_list)), np.float) # visibility of each joint
			train_bbox_pair = np.zeros((batch_size, 4, 4), dtype = np.float)
			i = 0
			while i < batch_size:
				if sample_set == 'train':
					img_pair_name = random.choice(self.img_pair)
				for index, name in enumerate(img_pair_name):
					joints = self.data_dict[name]['joints']
					box = self.data_dict[name]['box']
					weight = np.asarray(self.data_dict[name]['weights'])
					train_weights[index] = weight

					img = self.open_img(name)
					# Create the bounding box according to joints
					# Notice cbox[0][1] is the center of the img
					# and cbox[2][3] is the crop img size
					padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp = 0.1)
					train_bbox[index] = cbox
					# joints location with relative representations
					new_j = self._relative_joints(cbox, padd, joints, to_size=64)
					gt_heatmap = self._generate_hm(64, 64, new_j, 64, weight) # size of gt_heatmap = 64*64*9
					img = self._crop_img(img, padd, cbox)

					img = img.astype(np.uint8)
					img = scm.imresize(img, (256,256)) # 256,256,3
					#img, gt_heatmap = self._augment(img, gt_heatmap)
					gt_heatmap = np.expand_dims(gt_heatmap, axis = 0)
					gt_heatmap = np.repeat(gt_heatmap, stacks, axis = 0)# convert to 4*64*64*9
					# use stack = 4 for intermediate supervision
					if normalize:
						train_img[index] = img.astype(np.float) / 255
					else :
						train_img[index] = img.astype(np.float) # 4(natch_size)*256*256*3(RGB)
					train_gtmap[index] = gt_heatmap # 4(batch_size)*4(nStack)*64*64*9(joints number)
				train_weights_pair[i] = train_weights
				train_img_pair[i] = train_img
				train_gtmap_pair[i] = train_gtmap
				train_bbox_pair[i] = train_bbox
				i = i + 1
				'''
				except :
					print(' [!] Error file: ', name)
				'''
			yield train_img_pair, train_gtmap_pair, train_weights_pair, train_bbox_pair # It's really intelligent to use yield and next!!
			
	def generator(self, batchSize = 4, stacks = 4, norm = True, sample = 'train'):
		""" Create a Sample Generator
		Args:
			batchSize 	: Number of image per batch 
			stacks 	 	: Stacks in HG model
			norm 	 	 	: (bool) True to normalize the batch
			sample 	 	: 'train'/'valid' Default: 'train'
		"""
		return self._aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample)
	
	# unused		
	def _augment(self,img, hm, max_rotation = 30):
		""" # IMPLEMENT DATA AUGMENTATION 
		"""
		if random.choice([0,1]): 
			r_angle =random.randint(-1*max_rotation, max_rotation)
			# img = transform.rotate(img, r_angle, preserve_range = True)
			# hm = transform.rotate(hm, r_angle)
		return img, hm

	"""
	# ----------------------- Other utils ----------------------------------
	# ---------------------------- Debugger --------------------------------				
	"""
	# unused
	def plot_img(self, name, plot = 'cv2'):
		""" Plot an image
		Args:
			name	: Name of the Sample
			plot	: Library to use (cv2: OpenCV, plt: matplotlib)
		"""
		if plot == 'cv2':
			img = self.open_img(name, color = 'BGR')
			cv2.imshow('Image', img)
		elif plot == 'plt':
			img = self.open_img(name, color = 'RGB')
			plt.imshow(img)
			plt.show()
	
	# unused
	def test(self, toWait = 0.2):
		""" TESTING METHOD
		You can run it to see if the preprocessing is well done.
		Wait few seconds for loading, then diaporama appears with image and highlighted joints
		Use `Esc` to quit
		Args:
			toWait : In sec, time between pictures
		"""
		self._create_train_table()
		self._create_sets()
		for i in range(len(self.train_set)):
			img = self.open_img(self.train_set[i])
			w = self.data_dict[self.train_set[i]]['weights']
			padd, box = self._crop_data(img.shape[0], img.shape[1], self.data_dict[self.train_set[i]]['box'], self.data_dict[self.train_set[i]]['joints'], boxp= 0.0)
			new_j = self._relative_joints(box,padd, self.data_dict[self.train_set[i]]['joints'], to_size=256)
			rhm = self._generate_hm(256, 256, new_j,256, w)
			rimg = self._crop_img(img, padd, box)
			# See Error in self._generator
			#rimg = cv2.resize(rimg, (256,256))
			rimg = scm.imresize(rimg, (256,256))
			#rhm = np.zeros((256,256,16))
			#for i in range(16):
			#	rhm[:,:,i] = cv2.resize(rHM[:,:,i], (256,256))
			grimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
			cv2.imshow('image', grimg / 255 + np.sum(rhm,axis = 2))
			# Wait
			time.sleep(toWait)
			if cv2.waitKey(1) == 27:
				print('Ended')
				cv2.destroyAllWindows()
				break
	
	# unused
	def _crop(self, img, hm, padding, crop_box):
		""" Given a bounding box and padding values return cropped image and heatmap
		Args:
			img			: Source Image
			hm			: Source Heat Map
			padding	: Padding
			crop_box	: Bounding Box
		"""
		img = np.pad(img, padding, mode = 'constant')
		hm = np.pad(hm, padding, mode = 'constant')
		max_lenght = max(crop_box[2], crop_box[3])
		img = img[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght //2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght //2]
		hm = hm[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght//2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
		return img, hm

	# ------------------------------- PCK METHODS-------------------------------
	# unused
	def pck_ready(self, idlh = 3, idrs = 12, testSet = None):
		""" Creates a list with all PCK ready samples
		(PCK: Percentage of Correct Keypoints)
		"""
		id_lhip = idlh
		id_rsho = idrs
		self.total_joints = 0
		self.pck_samples = []
		for s in self.data_dict.keys():
			if testSet == None:
				if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1:
					self.pck_samples.append(s)
					wIntel = np.unique(self.data_dict[s]['weights'], return_counts = True)
					self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
			else:
				if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1 and s in testSet:
					self.pck_samples.append(s)
					wIntel = np.unique(self.data_dict[s]['weights'], return_counts = True)
					self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
		print('PCK PREPROCESS DONE: \n --Samples:', len(self.pck_samples), '\n --Num.Joints', self.total_joints)

	# unused
	def getSample(self, sample = None):
		""" Returns information of a sample
		Args:
			sample : (str) Name of the sample
		Returns:
			img: RGB Image
			new_j: Resized Joints 
			w: Weights of Joints
			joint_full: Raw Joints
			max_l: Maximum Size of Input Image
		"""
		if sample != None:
			try:
				joints = self.data_dict[sample]['joints']
				box = self.data_dict[sample]['box']
				w = self.data_dict[sample]['weights']
				img = self.open_img(sample)
				padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp = 0.2)
				new_j = self._relative_joints(cbox,padd, joints, to_size=256)
				joint_full = np.copy(joints)
				max_l = max(cbox[2], cbox[3])
				joint_full = joint_full + [padd[1][0], padd[0][0]]
				joint_full = joint_full - [cbox[0] - max_l //2,cbox[1] - max_l //2]
				img = self._crop_img(img, padd, cbox)
				img = img.astype(np.uint8)
				img = scm.imresize(img, (256,256))
				return img, new_j, w, joint_full, max_l
			except:
				return False
		else:
			print('Specify a sample name')
