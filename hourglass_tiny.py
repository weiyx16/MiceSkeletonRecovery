# -*- coding: utf-8 -*-
"""
Deep Mice Pose Estimation Using Stacked Hourglass Network

Project by @Eason
Adapted from @Walid Benbihi [source code]github : https://github.com/wbenbihi/hourglasstensorlfow/

---
Model and Training function
---
"""

import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
import cv2
from tqdm import tqdm
from skimage.transform import ProjectiveTransform, warp

debug = False

class HourglassModel():
	""" 
	HourglassModel class: (to be renamed)
	Generate TensorFlow model to train and predict Mice Pose from images
	Please check README.txt for further information on model management.
	"""
	def __init__(self, gpu_frac = 0.75, nFeat = 256, nStack = 4, nModules = 1, nLow = 4, outputDim = 4, batch_size = 4, 
		drop_rate = 0.2, lear_rate = 2.5e-4, decay = 0.96, decay_step = 100, dataset = None, training = True, 
		w_summary = False, logdir_train = None, logdir_test = None, tiny = True,
		w_loss = False, name = 'mice_tiny_hourglass', model_save_dir = None, joints = ['nose','r_ear','l_ear','tail_base']):
		""" Initializer
		Args:
			nStack				: number of stacks (stage/Hourglass modules)
			nFeat				: number of feature channels on conv layers
			nLow				: number of downsampling (pooling) per module
			outputDim			: number of output Dimension (9 for full mice data)
			batch_size			: size of training/testing Batch
			dro_rate			: Rate of neurons disabling for Dropout Layers
			lear_rate			: Learning Rate starting value
			decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
			decay_step			: Step to apply decay
			dataset				: Dataset (class DataGenerator)
			training			: (bool) True for training / False for prediction
			w_summary			: (bool) True/False for summary of weight (to visualize in Tensorboard) (set false)
			w_loss				: (bool) used to weighted loss (didn't calculate loss on unvisible joints)
			tiny				: (bool) Activate Tiny Hourglass
			name				: name of the model
			Joints				: Full one (9 points) ['nose','r_ear','l_ear','rf_leg','lf_leg','rb_leg','lb_leg','tail_base','tail_end']
		"""
		self.nStack = nStack
		self.nFeat = nFeat
		self.nModules = nModules
		self.outDim = outputDim
		self.batchSize = batch_size
		self.training = training
		self.w_summary = w_summary
		self.tiny = tiny
		self.dropout_rate = drop_rate
		self.learning_rate = lear_rate
		self.decay = decay
		self.name = name
		self.decay_step = decay_step
		self.nLow = nLow
		self.dataset = dataset
		self.cpu = '/cpu:0'
		self.gpu = '/gpu:0'
		self.logdir_train = logdir_train
		self.logdir_test = logdir_test
		self.model_save_dir = model_save_dir
		self.joints = joints
		self.w_loss = w_loss
		self.gpu_frac = gpu_frac
		self.saver = None
		self.lambda_geo = 100
	"""
	# ---------------- Self-Parameters Accessor --------------
	"""
	def get_input(self):
		""" Returns Input (Placeholder) Tensor
		Image Input :
			Shape: (None, 256, 256, 3)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
		return self.img

	def get_output(self):
		""" Returns Output Tensor
		Output Tensor :
			Shape: (None, nbStacks, 64, 64, outputDim)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
		return self.output

	def get_label(self):
		""" Returns Label/Ground Truth Map (Placeholder) Tensor
		Image Input :
			Shape: (None, nbStacks, 64, 64, outputDim)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
		return self.gtMaps

	def get_loss(self):
		""" Returns Loss Tensor
		Image Input :
			Shape: (1,)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
		return self.loss

	def get_saver(self):
		""" Returns Saver
		USE ONLY IF YOU KNOW WHAT YOU ARE DOING
		Warning:
			Be sure to build the model first
		"""
		return self.saver
	
	"""
	# ---------------- Model Graph Generator --------------	
	"""
	def generate_model(self):
		""" 
			Create the complete `model graph`
			Including the Network model/ Loss&Optimizer/Accuracy/ Some visualization params
		"""
		startTime = time.time()

		# create the model on GPU with inputs/Model Graph/loss
		with tf.device(self.gpu):
			with tf.name_scope('inputs'):
				# Shape Input Image - batchSize: None, camera_view: 4, height: 256, width: 256, channel: 3 (RGB) in NHWC format
				# NOTICE Set 256 unchanged
				self.img = tf.placeholder(dtype= tf.float32, shape= (None, 4, 256, 256, 3), name = 'input_img')
				if self.w_loss:
					self.weights = tf.placeholder(dtype = tf.float32, shape = (None, 4, self.outDim))
				# Shape Ground Truth Map: batchSize x camera_view x nStack x 64 x 64 x outDim
				# Intermediate supervision: so need multiple by nStack = 4				
				self.gtMaps = tf.placeholder(dtype = tf.float32, shape = (None, 4, self.nStack, 64, 64, self.outDim), name = 'groundtruth')
				self.bbox = tf.placeholder(dtype = tf.float32, shape = (None, 4, 4), name = 'input_crop_boundingbox')
			inputTime = time.time()
			print('-- Model Inputs : Done (' + str(int(abs(inputTime-startTime))) + ' sec.)')

			# Shape HG output: batchSize x nStack x 64 x 64 x outDim
			# But shape the whole model output is: batchSize x camera_view x nStack x 64 x 64 x outDim (self.output)
			# Generate the graph of the whole hourglass model here.
			self.output_list = []
			for i in range(4):
				self.output_list.append(self._graph_hourglass(self.img[:,i,:,:,:]))
				print('-- -- Model Graph for No. %d' %(i), ' View')
			self.output = tf.stack(self.output_list, axis=1)
			graphTime = time.time()
			print('-- Model Graph : Done (' + str(int(abs(graphTime-inputTime))) + ' sec.)')

			with tf.name_scope('loss'):
				# use sigmoid_cross_Ent to measure the loss
				if self.w_loss:
					self.gt_loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_ground_truth_loss')
				else:
					self.gt_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
				# Introduce reprojection loss 
				if self.training:
					self.geometry_loss = self.lambda_geo * tf.reduce_mean(self.camera_reproject_loss(), name='reduced_camera_reproject_loss')
				else:
					self.geometry_loss = self.gt_loss
				self.loss = tf.add(self.gt_loss, self.geometry_loss)
			lossTime = time.time()	
			print('-- Model Loss : Done (' + str(int(abs(lossTime-graphTime))) + ' sec.)')

		# create the model on CPU with Model Accuracy(in Eculidean space)&Learning rate
		with tf.device(self.cpu):
			with tf.name_scope('accuracy'):
				# Accurancy between network output and the gt-heatmap (in argmax location)
				# Is similar to loss but they are not computed in the same way
				self._accuracy_computation()
			accurTime = time.time()
			print('-- Model Accuracy : Done (' + str(int(abs(accurTime-lossTime))) + ' sec.)')

			# the learning rate and its decay
			with tf.name_scope('steps'):
				self.train_step = tf.Variable(0, name = 'global_step', trainable= False)
			with tf.name_scope('lr'):
				self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay, staircase= True, name= 'learning_rate')
			lrTime = time.time()
			print('-- Model Learning rate & Steps: Done (' + str(int(abs(lrTime-accurTime))) + ' sec.)')
		
		# Q: why it didn't input the epoch number and the Iteration number
		# A: They are stored in the trainer not in the graph
		# create the model on GPU with Optimizer/Loss Minimizer
		with tf.device(self.gpu):
			# the optimizer method
			with tf.name_scope('rmsprop'):
				self.rmsprop = tf.train.RMSPropOptimizer(learning_rate= self.lr)
			optimTime = time.time()
			print('-- Model Optimizer : Done (' + str(int(abs(optimTime-lrTime))) + ' sec.)')

			with tf.name_scope('minimizer'):
				self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				# before run train_rmsprop-> run update_ops first
				with tf.control_dependencies(self.update_ops):
					self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
			minimTime = time.time()
			print('-- Model Loss Minimizer : Done (' + str(int(abs(minimTime-optimTime))) + ' sec.)')

		self.init = tf.global_variables_initializer()
		initTime = time.time()
		print('-- Model Params Initial : Done (' + str(int(abs(initTime-minimTime))) + ' sec.)')

		# create the model on CPU with the Trainer Visualization in TF board
		with tf.device(self.cpu):
			with tf.name_scope('training'):
				# use summary ops to show in the tensorboard
				tf.summary.scalar('gt_loss', self.gt_loss, collections = ['train'])
				tf.summary.scalar('geometry_loss', self.geometry_loss, collections = ['train'])
				tf.summary.scalar('loss', self.loss, collections = ['train'])
				tf.summary.scalar('learning_rate', self.lr, collections = ['train'])
			with tf.name_scope('summary'):
				for i in range(len(self.joints)):
					tf.summary.scalar(self.joints[i], self.joint_accur[i], collections = ['train', 'test'])
		
		with tf.device(self.cpu):
			self.saver = tf.train.Saver() #, keep_checkpoint_every_n_hours=2)
		
		with tf.device(self.gpu):
			self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
			self.test_summary = tf.summary.FileWriter(self.logdir_test)
			#self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph()) # don't write down the summary of weighs for now
		
		summTime = time.time()
		print('-- Model Saver & Summary : Done (' + str(int(abs(summTime-initTime))) + ' sec.)')
		# use merge_all to (use to merge all ops/scalar/histogram to save in the train collection)
		# E.g. use: tf.summary.FileWriter(log_dir).add_summary(self.train_op or self.weight_op, epoch*epochSize + i)
		self.train_op = tf.summary.merge_all('train')
		# self.test_op = tf.summary.merge_all('test') # summary merge if for validation
		self.weight_op = tf.summary.merge_all('weight') # used for conv function
		endTime = time.time()
		print('>>>>> Model created in (' + str(int(abs(endTime-startTime))) + ' sec.)')
		# every time is an object and if we don't need them then remove them
		del endTime, startTime, initTime, optimTime, minimTime, lrTime, accurTime, lossTime, graphTime, inputTime, summTime		
	
	def weighted_bce_loss(self):
		""" Create Weighted Loss Function
		Don't calculate loss on unlabel joint (which unvisibility and has a blank heatmap)
		"""

		'''
		# We can use: self.weights + tf.expand_dims to remove empty heatmap
		self.bceloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
		e1 = tf.expand_dims(self.weights, axis = 1, name = 'expdim01')
		e2 = tf.expand_dims(e1, axis = 1, name = 'expdim02')
		e3 = tf.expand_dims(e2, axis = 1, name = 'expdim03')
		return tf.multiply(e3, self.bceloss, name = 'lossW')
		'''
		loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps)
		weights = tf.reduce_sum(self.gtMaps, axis=[3, 4], keepdims=True)
		return tf.reduce_sum(weights * loss, axis=[1, 2, 5], keepdims=True) # shape is batchsize*1*1*64*64*1 (sum over view+stack+joints)

	def camera_reproject_loss(self):
		"""
			Create camera geometry placement loss for we use multi-camera(4 cameras)
			Introduce epipolar constraint on the output heatmap between difference views
			ref: Monet: multiview semi-supervised keypoint via epipolar divergence
			Sum up for all the view-pair&stack&joints number
		"""
		
		# Don't forget to use weights = tf.reduce_sum(self.gtMaps, axis=[2, 3], keepdims=True)
		# batchSize x camera_view x nStack x 64 x 64 x outDim
		output_shape = self.output.get_shape().as_list()
		self.project_loss = []
		for batch_index in range(self.batchSize):
			# Batch_size
			# First step: warp the heatmaps by the Hh matrix	
			# Second step: calculate (a,b) and the loss for this frame
			cur_project_loss = tf.zeros([1,])
			camera_pair = [[0,1],[0,2],[1,3],[2,3]]
			for pair_index in range(4):
				camera_view_i = camera_pair[pair_index][0]
				camera_view_j = camera_pair[pair_index][1]
				# In sum: 4 pairs
				# warp matrix Hh
				# Hh is defined on two-paired heatmaps (and its coordinate)
				# but the Hr is defined on two-paired source size image (and its coordinate)
				Hr_i = self.dataset.Rectification_Homography_Matrix[pair_index][0]
				self.Hh_i_matrix = self._warp_matrix(self.bbox[batch_index,camera_view_i,:], Hr_i)
				Hr_j = self.dataset.Rectification_Homography_Matrix[pair_index][1]
				self.Hh_j_matrix = self._warp_matrix(self.bbox[batch_index,camera_view_j,:], Hr_j)
				#for stack_index in range(output_shape[2]):
				# Only calculate on the final stack for now
				for joint_index in range(output_shape[5]):
					# Each_joints
					# warp images 
					# test_matrix = np.eye(3); test_matrix[0,2] = 10; test_matrix = tf.convert_to_tensor(test_matrix)	
					self.heatmap_src_i = self.output[batch_index,camera_view_i, output_shape[2]-1, :,:,joint_index]
					# self.heatmap_warp_i = warp(np.asarray(self.heatmap_src_i), np.asarray(tf.matrix_inverse(Hh_i_matrix)))
					self.heatmap_warp_i = self._warp_img(self.heatmap_src_i, self.Hh_i_matrix)
					self.heatmap_src_j = self.output[batch_index,camera_view_j, output_shape[2]-1, :,:,joint_index]
					# self.heatmap_warp_j = warp(np.asarray(self.heatmap_src_j), np.asarray(tf.matrix_inverse(Hh_j_matrix)))
					self.heatmap_warp_j = self._warp_img(self.heatmap_src_j, self.Hh_j_matrix)							
					
					# print('For current pair and joint %d image pairs warp done' %(joint_index))
					# a, b: for every v in warped image i: av+b is coorsponding row in warped image j
					f_yi = self.dataset.Intrinsic[camera_view_i][1,1]
					f_yj = self.dataset.Intrinsic[camera_view_j][1,1]
					p_yi = self.dataset.Intrinsic[camera_view_i][1,2]
					p_yj = self.dataset.Intrinsic[camera_view_j][1,2]
					a = tf.cast(tf.div(f_yj, f_yi), 'float32')
					b = tf.cast(tf.add(p_yj, tf.multiply(tf.div(-f_yj, f_yi), p_yi)), 'float32')
					# Qi, Qj_i
					v = tf.linspace(0.0, output_shape[4]-1, output_shape[4])
					a = tf.add(tf.zeros_like(v), a)
					v_j = tf.add(a*v, b)
					v_j = tf.cast(tf.floor(tf.clip_by_value(v_j, 0.0, output_shape[4]-1)), 'int32')
					v_j = tf.reshape(v_j, (v_j.shape[0], 1))
					# Notice add sigmoid on final output to avoid 0*log(0/c) a little different from source definition
					Q_hat_i = tf.nn.sigmoid(tf.reduce_max(self.heatmap_warp_i, axis = 1)) # rescale to (0~1)
					Q_hat_j = tf.nn.sigmoid(tf.reduce_max(tf.gather_nd(self.heatmap_warp_j, v_j), axis = 1))
					# Calculate loss
					cur_project_loss += tf.reduce_sum(-Q_hat_i*tf.log(tf.clip_by_value(tf.div_no_nan(Q_hat_i, Q_hat_j),1e-10,1.0)))
					#tf.distributions.kl_divergence(tf.distributions.Categorical(Q_hat_i), tf.distributions.Categorical(Q_hat_j), allow_nan_stats = False)
					#maybe your should normalization the Q to Sum(Q)==1 (a PDF)
					#or use softmax to convert the Q into a distritbution with probability
					#Or use softmax_cross_entropy_with_logits directly for someone prove that KL and SCE is only gapped by a const

			self.project_loss.append(cur_project_loss)
			print('-- -- Geometry Loss for No. %d' %(batch_index), ' batch')
		self.project_loss_all = tf.stack(self.project_loss, axis=0)
		return self.project_loss_all #tf.reduce_sum(self.project_loss, axis=[1, 2, 3], keepdims=True)		
	
	def _warp_matrix(self, bbox, Hr):
		"""
			Calculate the true warping matrix including the crop and resize of the input img to the output heatmap
			Because the Hr matrix is defined on the source img
			ref: Monet: multiview semi-supervised keypoint via epipolar divergence
		"""
	    # Test for new area
		bbox_src = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
		bbox_new_view = []
		for i in range(2):
			for j in range(2):
				corner = tf.expand_dims(tf.convert_to_tensor([bbox_src[i*2], bbox_src[2*j+1],1]), -1)
				corner_new = tf.matmul(tf.cast(tf.convert_to_tensor(Hr), 'float32'), corner)
				corner_new = corner_new / corner_new[2] # tf.div(corner_new[0], corner_new[2])
				bbox_new_view.append(tf.squeeze(tf.transpose(corner_new[:2], (1,0))))
		bbox_new_view = tf.stack(bbox_new_view, axis=0)
		bbox_new_view = tf.stack([tf.reduce_min(bbox_new_view, 0)[0], tf.reduce_min(bbox_new_view, 0)[1],tf.reduce_max(bbox_new_view, 0)[0], tf.reduce_max(bbox_new_view, 0)[1]])
		bbox_new_view = tf.stack([bbox_new_view[0], bbox_new_view[1], bbox_new_view[2] - bbox_new_view[0], bbox_new_view[3] - bbox_new_view[1]])
		s = 64 / tf.maximum(bbox[2], bbox[3]) #tf.cast(256 / tf.maximum(bbox[2], bbox[3]), 'float32') # input network cropped image size / source cropped image size
		Hb = tf.convert_to_tensor([[s, 0, -s*bbox[0]],
									[0, s, -s*bbox[1]],
									[0, 0, 1]])
		Hb_inv = tf.cast(tf.matrix_inverse(Hb), 'float32')
		s_2 = 64 / tf.maximum(bbox_new_view[2], bbox_new_view[3])
		Hb_hat = tf.convert_to_tensor([[s_2, 0, -s_2*bbox_new_view[0]],
										[0, s_2, -s_2*bbox_new_view[1]],
										[0, 0, 1.0]])
		Hb_hat = tf.cast(Hb_hat, 'float32')

		return tf.matmul(tf.matmul(Hb_hat,tf.cast(tf.convert_to_tensor(Hr), 'float32')), Hb_inv)

	def _warp_img(self, heatmap_src, Hh):
		"""
			Inverse warping by Matrix Hh through bilinear interpolation
			heatmap_warp(x) = heatmap_src(Hh^-1 * x)
			ref: Spatial Transformer Networks NIPS 2015 (include warping as part of the network with parameter to be learned)
			# we don't need to learn the warp parameters here
			# Notice the warping matrix is a little different from the one on pixel:
			[Matrix] * [x,y,1] -> [x',y',1]
			But in interpolate function, use grid to stand for the pixel location and [0,63] -> linespace[0,63]
		"""
		# the input and output can have different size
		out_height = tf.shape(heatmap_src)[0]
		out_width = tf.shape(heatmap_src)[1]
		Hh_inv = tf.cast(tf.matrix_inverse(Hh), 'float32')
		# grid of (x_t, y_t, 1) (target) in ref
		grid = self._meshgrid(out_height, out_width)
		# Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
		T_g = tf.matmul(Hh_inv, grid)
		T_g = T_g / T_g[2]
		x_s_flat = T_g[0,:]
		y_s_flat = T_g[1,:]
		heatmap_warp = tf.expand_dims(tf.reshape(self._interpolate(heatmap_src, x_s_flat, y_s_flat), tf.stack([out_height, out_width])), -1)
		return heatmap_warp

	def _meshgrid(self, height, width):
		with tf.variable_scope('warping_meshgrid'):
			# This should be equivalent to:
			#  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
			#                         np.linspace(-1, 1, height))
			#  ones = np.ones(np.prod(x_t.shape))
			#  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
			
			x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
			                tf.transpose(tf.expand_dims(tf.linspace(0.0, tf.cast(width, 'float32') - tf.constant(1.0), width), 1), [1, 0])) # 64*1 * 1*64
			y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, tf.cast(height, 'float32') - tf.constant(1.0), height), 1),
			                tf.ones(shape=tf.stack([1, width]))) # 64*1 * 1*64
			# y_t: [1...1]      x_t: [1 ..0.. -1]
			#      .0...0.            . ..0.. .
			#      [-1...-1]		 [1 ..0.. -1]
			x_t_flat = tf.reshape(x_t, (1, -1))
			y_t_flat = tf.reshape(y_t, (1, -1))
			ones = tf.ones_like(x_t_flat)
			grid = tf.concat([x_t_flat, y_t_flat, ones], 0)
			return grid

	def _interpolate(self, heatmap_src, x, y):
		'''
			# Bilinear interpolation by grid methods
		'''
		with tf.variable_scope('_interpolate'):
			# constants
			height_f = tf.cast(tf.shape(heatmap_src)[0], 'float32')
			width_f = tf.cast(tf.shape(heatmap_src)[1], 'float32')
			width = tf.cast(width_f, 'int32')
			height = tf.cast(height_f, 'int32')
			x = tf.cast(x, 'float32')
			y = tf.cast(y, 'float32')
			
			# bilinear sampling
			x0 = tf.cast(tf.floor(x), 'int32')
			x1 = x0 + 1
			y0 = tf.cast(tf.floor(y), 'int32')
			y1 = y0 + 1
			
			x0 = tf.clip_by_value(x0, 0, width - 1)
			x1 = tf.clip_by_value(x1, 0, width - 1)
			y0 = tf.clip_by_value(y0, 0, height - 1)
			y1 = tf.clip_by_value(y1, 0, height - 1)

			# calculate interpolated values with weights
			x0_f = tf.cast(x0, 'float32')
			x1_f = tf.cast(x1, 'float32')
			y0_f = tf.cast(y0, 'float32')
			y1_f = tf.cast(y1, 'float32')
			
			# Use 1-D index instead of 2-D index, so take the width into consideration
			base_y0 = y0 * width
			base_y1 = y1 * width
			idx_a = base_y0 + x0
			idx_b = base_y1 + x0
			idx_c = base_y0 + x1
			idx_d = base_y1 + x1
			# use indices to lookup pixels in the flat image and restore
			im_flat = tf.cast(tf.reshape(heatmap_src, [-1]), 'float32')
			Ia = tf.expand_dims(tf.gather(im_flat, idx_a), 1)
			Ib = tf.expand_dims(tf.gather(im_flat, idx_b), 1)
			Ic = tf.expand_dims(tf.gather(im_flat, idx_c), 1)
			Id = tf.expand_dims(tf.gather(im_flat, idx_d), 1)

			wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
			wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
			wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
			wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
			output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
			return output

	"""
	# --------------- Model Training --------------
	"""
	def _train(self, nEpochs = 100, epochSize = 100, saveStep = 20, validIter = 10):
		"""
			`Real training process`
		"""
		print(">>>>> Begin training!")
		with tf.name_scope('Train'):
			self.generator = self.dataset.generator(self.batchSize, self.nStack, norm = True, sample = 'train')
			startTime = time.time()
			self.resume = {}
			# self.resume['accur'] = [] # for validation
			# self.resume['err'] = []
			self.resume['loss'] = []
			
			min_cost = 1e5
			for epoch in tqdm(range(nEpochs), ncols=70):
				epochstartTime = time.time()
				avg_cost = 0.
				cost = 0.

				# Training Part
				for i in range(epochSize):
					img_train, gt_train, weight_train, bbox_train = next(self.generator)
					# the saveStep is the step to save summary not the model
					if i % saveStep == saveStep - 1:
						if self.w_loss:
							_, cur_loss, cur_geometry_loss, gt_loss, train_summary = self.Session.run([self.train_rmsprop, self.loss, self.geometry_loss, self.gt_loss, self.train_op], 
								feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train, self.bbox: bbox_train})
						else:
							_, cur_loss, cur_geometry_loss, gt_loss, train_summary = self.Session.run([self.train_rmsprop, self.loss, self.geometry_loss, self.gt_loss, self.train_op], 
								feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.bbox: bbox_train })
						# Save summary (Loss + Accuracy)
						# FileWriter to logdir_train
						print(' [**] Saving summary here...')
						self.train_summary.add_summary(train_summary, epoch*epochSize + i)
						self.train_summary.flush()
						
					else:
						if self.w_loss:
							# DEBUGGER
							if debug:
								_, cur_loss, cur_geometry_loss, gt_loss, Hh_i_matrix, Hh_j_matrix, heatmap_src_i, heatmap_src_j, heatmap_warp_i, heatmap_warp_j = \
									self.Session.run([self.train_rmsprop, self.loss, self.geometry_loss, self.gt_loss, self.Hh_i_matrix, self.Hh_j_matrix, 
										self.heatmap_src_i, self.heatmap_src_j, self.heatmap_warp_i, self.heatmap_warp_j], 
									feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train, self.bbox: bbox_train})
							else:
								_, cur_loss, cur_geometry_loss, gt_loss = self.Session.run([self.train_rmsprop, self.loss, self.geometry_loss, self.gt_loss], 
									feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train, self.bbox: bbox_train})
						else:
							
							# DEBUGGER
							if debug:
								_, cur_loss, cur_geometry_loss, gt_loss, Hh_i_matrix, Hh_j_matrix, heatmap_src_i, heatmap_src_j, heatmap_warp_i, heatmap_warp_j = \
									self.Session.run([self.train_rmsprop, self.loss, self.geometry_loss, self.gt_loss, self.Hh_i_matrix, self.Hh_j_matrix, 
										self.heatmap_src_i, self.heatmap_src_j, self.heatmap_warp_i, self.heatmap_warp_j], 
									feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.bbox: bbox_train})
							else:
								_, cur_loss, cur_geometry_loss, gt_loss = self.Session.run([self.train_rmsprop, self.loss, self.geometry_loss, self.gt_loss], 
									feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.bbox: bbox_train})

					cost += cur_loss
					avg_cost += cur_loss/epochSize
					print(' [*] In Epoch {}, Loop {}, geometry loss is {}, ground truth loss is {}, total loss is {}'.format(epoch, i, cur_geometry_loss, gt_loss, cur_loss))
				epochfinishTime = time.time()
				
				# Save Weight (axis = epoch) for all the conv
				if self.w_loss:
					weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train, self.bbox: bbox_train})
				else :
					weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train, self.bbox: bbox_train})
				
				self.train_summary.add_summary(weight_summary, epoch)
				self.train_summary.flush()
				
				print('\n-- Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(int(epochfinishTime-epochstartTime)) + ' sec.\n'
					 + ' - time_per_batch: ' + str(((epochfinishTime-epochstartTime)/epochSize))[:4] + ' sec.', ' - cost_per_batch: ' + str(avg_cost))

				if cost < 1.2 * min_cost:
					# Save model for each epoch
					with tf.name_scope('save'):
						self.saver.save(self.Session, os.path.join(self.model_save_dir, str(self.name)), global_step=epoch+1)
					min_cost = min(cost, min_cost)
				print('-- Saving new model with average cost: {}\n'.format(avg_cost))

				self.resume['loss'].append(cost)
				
				'''
				# Validation Part
				accuracy_array = np.array([0.0]*len(self.joint_accur))
				for i in range(validIter):
					img_valid, gt_valid, _ = next(self.generator)
					accuracy_pred = self.Session.run(self.joint_accur, feed_dict = {self.img : img_valid, self.gtMaps: gt_valid})
					accuracy_array += np.array(accuracy_pred, dtype = np.float) / validIter
				print('-- Avg_accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%' )
				self.resume['accur'].append(accuracy_pred)
				self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
				valid_summary = self.Session.run(self.test_op, feed_dict={self.img : img_valid, self.gtMaps: gt_valid})
				self.test_summary.add_summary(valid_summary, epoch)
				self.test_summary.flush()
				'''
			print('>>>>> Training Done')
			print('-- Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochSize * self.batchSize) )
			print('-- Final Loss: ' + str(cost) + '\n' + ' Loss Discimination: ' + str(100*self.resume['loss'][-1]/(self.resume['loss'][0] + 0.001)) + '%' )
			# print('-- Relative Accurancy Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) +'%')
			print('-- Training Time: ' + str(datetime.timedelta(seconds=time.time() - startTime)))
			
	def training_init(self, nEpochs = 100, epochSize = 100, saveStep = 20, valid_iter = 10, pre_trained = None):
		""" Initialize the training process (And into _train():the true training function)

		Args:
			nEpochs			: Number of Epochs to train
			epochSize		: Size of one Epoch
			saveStep		: Step to save 'train' summary (has to be lower than epochSize)
			valid_iter		: Step to apply validation steps
			pre_trained			: Pre-trained Model to load (None if training from scratch) (see README for further information)
		"""
		print(">>>>> Begin setting the training process!")
		with tf.name_scope('Session'):
			with tf.device(self.gpu):
				self._init_weight()
				self._define_saver_summary()
				if pre_trained is not None:
					try:
						print('-- Loading Pre-trained Model')
						load_t = time.time()
						'''
						ckpt = tf.train.get_checkpoint_state(pre_trained)
						if ckpt and ckpt.model_checkpoint_path:
							self.saver.restore(self.Session, ckpt.model_checkpoint_path)
						'''
						self.saver.restore(self.Session, pre_trained)
						print('-- Pre-trained Model Loaded (', time.time() - load_t,' sec.)')
						del load_t
					except Exception:
						print('-- Pre-trained model Loading Failed!')
				self._train(nEpochs, epochSize, saveStep, validIter=valid_iter)
		
	def _define_saver_summary(self, summary = True):
		""" Check Summary and Saver directory exists or not
		Args:
			logdir_train	: Path to train summary directory
			logdir_test		: Path to test summary directory
		"""
		if (self.logdir_train == None) or (self.logdir_test == None):
			raise ValueError('Train/Test directory not assigned')
			
	def _init_weight(self):
		""" Initialize weights and session
		"""
		print('-- Session weights initialization')
		self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = self.gpu_frac)
		self.Session = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
		t_start = time.time()
		self.Session.run(self.init) # initialize the parameters in session
		print('-- Session weights initialized in ' + str(int(time.time() - t_start)) + ' sec.')
		del t_start

	"""
	# --------------- Model Eavluation --------------
	"""
	def generate_model_eval(self):
		""" 
			Create the complete `model graph`
			Including the Network model/ Loss&Optimizer/Accuracy/ Some visualization params
			Useful in evaluation
		"""
		startTime = time.time()

		# create the model on GPU with inputs/Model Graph/loss
		with tf.device(self.gpu):
			with tf.name_scope('inputs'):
				# Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB) in NHWC format
				# NOTICE Set 256 unchanged
				self.img = tf.placeholder(dtype= tf.float32, shape= (None, 256, 256, 3), name = 'input_img')
				if self.w_loss:
					self.weights = tf.placeholder(dtype = tf.float32, shape = (None, self.outDim))
				# Shape Ground Truth Map: batchSize x camera_view x nStack x 64 x 64 x outDim
				# Intermediate supervision: so need multiple by nStack = 4				
				self.gtMaps = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 64, 64, self.outDim), name = 'groundtruth')
				self.bbox = tf.placeholder(dtype = tf.float32, shape = (None, 4), name = 'input_crop_boundingbox')
			inputTime = time.time()
			print('-- Model Inputs : Done (' + str(int(abs(inputTime-startTime))) + ' sec.)')

			self.output = self._graph_hourglass(self.img)
			graphTime = time.time()
			print('-- Model Graph : Done (' + str(int(abs(graphTime-inputTime))) + ' sec.)')

		self.init = tf.global_variables_initializer()
		initTime = time.time()
		print('-- Model Params Initial : Done (' + str(int(abs(initTime-graphTime))) + ' sec.)')
		
		with tf.device(self.cpu):
			self.saver = tf.train.Saver() #, keep_checkpoint_every_n_hours=2)
		
		endTime = time.time()
		print('>>>>> Model created in (' + str(int(abs(endTime-startTime))) + ' sec.)')
		# every time is an object and if we don't need them then remove them
		del endTime, startTime, initTime, graphTime, inputTime		

	def restore(self, pre_trained = None):
		""" Restore a pretrained model (`During evaluation`)
			Args:
			load	: Model to load (None if training from scratch) (see README for further information)
		"""
		with tf.name_scope('Session'):
			with tf.device(self.gpu):
				self._init_session()
				self._define_saver_summary(summary = False)
				if pre_trained is not None:
					print('-- Loading Trained Model')
					t = time.time()
					self.saver.restore(self.Session, pre_trained)
					print('-- Model Loaded (', time.time() - t,' sec.)')
				else:
					print('Please give a Model in args (see README for further information)')

	def _init_session(self):
		""" Initialize Session
		"""
		print('-- Session initialization')
		t_start = time.time()
		self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = self.gpu_frac)
		self.Session = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
		print('-- Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')
		del t_start
		
	"""
	# ---------- Network Structure -------------
	"""	

	def _graph_hourglass(self, inputs):
		""" Create the `Network Graph(Stacked Hourglass)`

			Args:
				inputs : TF Tensor (placeholder) of shape (None, 3, 256, 256) (The size of self.img)
		"""
		with tf.variable_scope('model') as scope:
			# preprocess the image
			with tf.variable_scope('preprocessing'):
				# Input Dim : batchsize x 256 x 256 x 3
				inputs = tf.transpose(inputs, [0,3,1,2]) # suitable to NCHW format
				pad1 = tf.pad(inputs, [[0,0],[0,0],[2,2],[2,2]], name='pad_1')
				# Dim pad1 : batchsize x 260 x 260 x 3
				# Filter size is the kernel numbers
				conv1 = self._conv_bn_relu(pad1, filters= 64, kernel_size = 6, strides = 2, name = 'conv_256_to_128')
				# Validing size: ceil(float(in_size-filter_size+1))/float(strides) 				
				# Dim conv1 : batchsize x 128 x 128 x 64 (128 = ceil(260-6+1)/2)
				# Why in source code use kernel_size = 7?

				r1 = self._residual(conv1, numOut = 128, name = 'r1')
				# Dim r1 : batchsize x 128 x 128 x 128 (channel from 64 to 128)
				pool1 = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding='VALID', data_format='NCHW')
				# Dim pool1 : batchsize x 64 x 64 x 128

				if self.tiny:
					r3 = self._residual(pool1, numOut=self.nFeat, name='r3')
					# Dim r3 : batchsize x 64 x 64 x self.nFeat(actually 256)
				else:
					r2 = self._residual(pool1, numOut= int(self.nFeat/2), name = 'r2')
					# Dim r2 : batchsize x 64 x 64 x self.nFeat/2
					r3 = self._residual(r2, numOut= self.nFeat, name = 'r3')
					# Dim r3 : batchsize x 64 x 64 x self.nFeat(actually 256)

			# Storage Table
			hg = [None] * self.nStack
			ll = [None] * self.nStack
			ll_ = [None] * self.nStack
			drop = [None] * self.nStack
			out = [None] * self.nStack
			out_ = [None] * self.nStack
			sum_ = [None] * self.nStack
			if self.tiny:
				with tf.variable_scope('stacks'):
					for i in range(0, self.nStack):
						with tf.variable_scope('stage_' + str(i)):
							if i == 0:
								hg[i] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
							else:
								hg[i] = self._hourglass(sum_[i-1], self.nLow, self.nFeat, 'hourglass')
							drop[i] = tf.layers.dropout(hg[i], rate = self.dropout_rate, training = self.training, name = 'dropout')
							ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, name= 'll')
							out[i] = self._conv(ll[i], self.outDim, 1, 1, 'out') # Output level i
							if i < self.nStack - 1:  # We don't need out_ and sum_ for the last stack (stack 3)
								out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'out_')
								if i == 0:
									sum_[i] = tf.add_n([out_[i], ll[i], r3], name= 'merge') 
									# notice out_ is not the real output, it's used to create the input pf next level
								else:
									sum_[i] = tf.add_n([out_[i], ll[i], sum_[i-1]], name= 'merge')
									# all of the three components have self,nFeat=256 channels (a little different from the paper)
					for i in range(0, self.nStack):
						out[i] = tf.transpose(out[i], [0,2,3,1])
				return tf.stack(out, axis = 1 , name = 'final_output') # out size =  batchsize * nstack * 64 * 64 * 9
				# stack: cascade the matrix from batchsize * 64 * 64 * 9 -> batchsize * nstack * 64 * 64 * 9 (put stacknumber in axis=1)
				# So it can add in intermediate superview on output of each stack

			else:
				# Full 4-Rank Houglass network 
				with tf.variable_scope('stacks'):
					for i in range(0, self.nStack):
						with tf.variable_scope('stage_' + str(i)):
							if i == 0:
								hg[i] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
							else:
								hg[i] = self._hourglass(sum_[i-1], self.nLow, self.nFeat, 'hourglass')
							drop[i] = tf.layers.dropout(hg[i], rate = self.dropout_rate, training = self.training, name = 'dropout')
							ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, name= 'conv')
							out[i] = self._conv(ll[i], self.outDim, 1, 1, 'out') # Output level i
							if i < self.nStack - 1:  # We don't need out_ and sum_ for the last stack (stack 3)
								ll_[i] =  self._conv(ll[i], self.nFeat, 1, 1, name = 'll')
								out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'out_')
								if i == 0:
									sum_[i] = tf.add_n([out_[i], ll_[i], r3], name= 'merge') 
									# notice out_ is not the real output, it's used to create the input pf next level
								else:
									sum_[i] = tf.add_n([out_[i], ll_[i], sum_[i-1]], name= 'merge')
									# all of the three components have self,nFeat=256 channels (a little different from the paper)
					for i in range(0, self.nStack):
						out[i] = tf.transpose(out[i], [0,2,3,1])
				return tf.stack(out, axis= 1 , name = 'final_output') # out size =  batchsize * nstack * 64 * 64 * 9

	"""
	# ---------- Network Module Units -------------
	"""	

	def _conv(self, inputs, filters, kernel_size = 1, strides = 1, name = 'conv'):
		""" Spatial Convolution (2D) `Singel conv2d`
		Args:
			inputs			: Input Tensor (Data Type : NCHW)
			filters			: Number of filters (output channels)
			kernel_size		: Size of kernel
			strides			: Stride
			name			: Name of the block
		Returns:
			conv			: Output Tensor (Convolved Input)
		"""
		with tf.variable_scope(name) as scope:
			# Kernel for convolution, Xavier Initialisation
			# initialize the kernel weight using xavier method
			try:
				kernel = tf.get_variable(name+'_weights', initializer = tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[1], filters]))
			except ValueError:
				# print('reuse variables here')
				scope.reuse_variables()
				kernel = tf.get_variable(name+'_weights')
			
			conv = tf.nn.conv2d(inputs, kernel, [1,1,strides,strides], padding='VALID', data_format='NCHW')
			if self.w_summary:
				with tf.device('/cpu:0'):
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return conv
			
	def _conv_bn_relu(self, inputs, filters, kernel_size = 1, strides = 1, name = 'conv_bn_relu'):
		""" `Spatial Convolution (Singel conv2d) + BatchNormalization + ReLU Activation`
		
		Args:
			inputs			: Input Tensor (Data Type : NCHW)  
			filters			: Number of filters (channels)
			kernel_size		: Size of kernel
			strides			: Stride
			name			: Name of the block
		Returns:
			norm			: Output Tensor
		"""
		
		with tf.variable_scope(name) as scope:
			# initialize the kernel weight using xavier method
			try:
				kernel = tf.get_variable(name+'_weights', initializer = tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[1], filters]))
			except ValueError:
				scope.reuse_variables()
				kernel = tf.get_variable(name+'_weights')
			
			conv = tf.nn.conv2d(inputs, kernel, [1,1,strides,strides], padding='VALID', data_format='NCHW')
			norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training, data_format='NCHW', reuse=tf.AUTO_REUSE, scope=scope)
			if self.w_summary:
				with tf.device('/cpu:0'):
					# Adding a histogram summary makes it possible to visualize your data's distribution in TensorBoard
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return norm
		'''
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
			# initialize the kernel weight using xavier method
			kernel = tf.get_variable(name+'_weights', initializer = tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[1], filters]))
			conv = tf.nn.conv2d(inputs, kernel, [1,1,strides,strides], padding='VALID', data_format='NCHW')
			norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training, data_format='NCHW', reuse=tf.AUTO_REUSE, scope=scope)
			print(conv)
			print(norm)
			print(kernel)
			if self.w_summary:
				with tf.device('/cpu:0'):
					# Adding a histogram summary makes it possible to visualize your data's distribution in TensorBoard
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return norm
		'''
	def _conv_block(self, inputs, numOut, name = 'conv_block'):
		""" Convolutional Block `Cascaded conv2d used in Residual unit` (Main-stream)
		
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the block
		Returns:
			conv_3/conv	: Output Tensor
		"""
		if self.tiny:
			with tf.variable_scope(name) as scope:
				norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training, data_format='NCHW', reuse=tf.AUTO_REUSE, scope=scope)
				pad = tf.pad(norm, np.array([[0,0],[0,0],[1,1],[1,1]]), name= 'pad')
				conv = self._conv(pad, int(numOut), kernel_size=3, strides=1, name= 'conv')
				return conv
		else:
			with tf.variable_scope(name) as scope:
				# Standard convolution in the paper with kernel size [1 -> 3 -> 1]
				with tf.variable_scope(name+'norm_1'):
					norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training, data_format='NCHW', reuse=tf.AUTO_REUSE, scope=scope)
					conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, name= 'conv')
				with tf.variable_scope(name+'norm_2'):
					norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training, data_format='NCHW', reuse=tf.AUTO_REUSE, scope=scope)
					pad = tf.pad(norm_2, np.array([[0,0],[0,0],[1,1],[1,1]]), name= 'pad')
					conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, name= 'conv')
				with tf.variable_scope(name+'norm_3'):
					norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training, data_format='NCHW', reuse=tf.AUTO_REUSE, scope=scope)
					conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, name= 'conv')
				return conv_3
	
	def _skip_layer(self, inputs, numOut, name = 'skip_layer'):
		""" Skip Layer `Core of Residual unit` (Skip-stream)
		
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the bloc
		Returns:
			Tensor of shape (None, inputs.height, inputs.width, numOut)
		"""
		# if the channel size is equal to desired one then use inputs if not then conv it first
		with tf.variable_scope(name):
			if inputs.get_shape().as_list()[1] == numOut:
				return inputs
			else:
				conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
				return conv				
	
	def _residual(self, inputs, numOut, name = 'residual_block'):
		""" `Residual Unit` (combination of both the skip_layer and conv_block
		
		Args:
			inputs	: Input Tensor
			numOut	: Number of Output Features (channels)
			name	: Name of the block
			`Add on the conv2d and skip layer`
		"""
		with tf.variable_scope(name):
			convb = self._conv_block(inputs, numOut)
			skipl = self._skip_layer(inputs, numOut)
			# use add_n to sum a list of tensors(equal to +)
			return tf.add_n([convb, skipl], name = 'res_block')
	
	def _hourglass(self, inputs, n, numOut, name = 'hourglass'):
		""" `n+1 level Hourglass Module`
		
		Args:
			inputs	: Input Tensor
			n		: Number of downsampling step (level number)
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.variable_scope(str(n)+name):
			# Upper Branch
			up_1 = self._residual(inputs, numOut, name = 'up_1')
			# Lower Branch
			low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID', data_format='NCHW') # downsampling with max pooling
			# Valid in maxpooling: math.floor; Same: math.ceil
			low_1= self._residual(low_, numOut, name = 'low_1')
			
			if n > 0:
				low_2 = self._hourglass(low_1, n-1, numOut, name = 'low_2')
			else:
				low_2 = self._residual(low_1, numOut, name = 'low_2')
			
			low_3 = self._residual(low_2, numOut, name = 'low_3')
			# Use nearest_neighbor_interpole to upsample
			# 4-D tensors: [batch, channels, height, width]	
			low_3_shape = low_3.get_shape().as_list()
			up_2 = tf.image.resize_nearest_neighbor(tf.transpose(low_3, [0,2,3,1]), [low_3_shape[2]*2, low_3_shape[3]*2], name = 'upsampling') # upsampling with nearest neighbor interpolation		
			up_2 = tf.transpose(up_2, [0,3,1,2])	
			return tf.add_n([up_2, up_1], name='out_hg')
		
	"""
	# ---------- Model Accuracy Utils for Validation -------------
	"""	
	
	def _accuracy_computation(self):
		""" Computes accuracy tensor
			for each joint in each image of the batch(average in these images)
			Notice use the output of the last_stack
		"""
		self.joint_accur = []
		for i in range(len(self.joints)):
			# What's saved in self.nstack-1 is the last output of the network
			self.joint_accur.append(self._accur(self.output[:, :, self.nStack - 1, :, :,i], self.gtMaps[:, :, self.nStack - 1, :, :, i], self.batchSize))
	
	def _accur(self, pred, gtMap, num_image):
		""" Given a Prediction batch (pred) and a Ground Truth batch (gtMaps)
		for a single joint in each image-pair of the batch(average in these images)
		and sum over all the view
		returns one minus the mean distance.
		Args:
			pred		: Prediction Batch (shape = num_image x 64 x 64)
			gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
			num_image 	: (int) Number of images in batch
		Returns:
			(float)
		"""
		err = tf.to_float(0)
		for i in range(num_image):
			for view in range(4):
				err = tf.add(err, self._compute_err(pred[i][view], gtMap[i][view]))
			# err is the distance-> bigger err means lower accurancy-> we need to 1-err
		return tf.subtract(tf.to_float(1), err/(num_image*4))

	def _compute_err(self, u, v):
		""" Given 2 tensors compute the euclidean distance (L2) between maxima locations
		like gt-heatmap and output-heatmap
		for a single joint in a single image
		Args:
			u		: 2D - Tensor (Height x Width : 64x64 )
			v		: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			(float) : Distance (in [0,1])
		"""
		u_x,u_y = self._argmax(u)
		v_x,v_y = self._argmax(v)
		# 90.51 = 64 * square(2) (for normalization)
		# this is for accuracy, not for loss(which is MSE (mean square error) in source paper but sigmoid_cross_entropy_with_logits here)
		return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))), tf.to_float(90.51))

	def _argmax(self, tensor):
		""" `ArgMax location in a single heatmap`

		Args:
			tensor	: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			arg		: Tuple of max position
		"""
		resh = tf.reshape(tensor, [-1])
		argmax = tf.argmax(resh, 0) # find the max-confident value index
		# use // to get int divided by int -> int
		# argmax is a plain number and convert it into a location
		return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])
