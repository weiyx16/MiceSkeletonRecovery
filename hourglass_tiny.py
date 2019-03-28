# -*- coding: utf-8 -*-
"""
Deep Mice Pose Estimation Using Stacked Hourglass Network

Project by @Eason
Adapted from @Walid Benbihi [source code]github : https://github.com/wbenbihi/hourglasstensorlfow/

---
Model and Training function
---
"""

"""
	TODO: Maybe I need to calculate loss over joints number
"""
import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
from tqdm import tqdm

class HourglassModel():
	""" 
	HourglassModel class: (to be renamed)
	Generate TensorFlow model to train and predict Mice Pose from images
	Please check README.txt for further information on model management.
	"""
	def __init__(self, gpu_frac = 0.75, nFeat = 256, nStack = 4, nModules = 1, nLow = 4, outputDim = 9, batch_size = 4, 
		drop_rate = 0.2, lear_rate = 2.5e-4, decay = 0.96, decay_step = 100, dataset = None, training = True, 
		w_summary = True, logdir_train = None, logdir_test = None, tiny = True,
		w_loss = False, name = 'mice_tiny_hourglass', model_save_dir = None, joints = ['nose','r_ear','l_ear','rf_leg','lf_leg','rb_leg','lb_leg','tail_base','tail_end']):
		""" Initializer
		Args:
			nStack				: number of stacks (stage/Hourglass modules)
			nFeat				: number of feature channels on conv layers
			nLow				: number of downsampling (pooling) per module
			outputDim			: number of output Dimension (16 for MPII)
			batch_size			: size of training/testing Batch
			dro_rate			: Rate of neurons disabling for Dropout Layers
			lear_rate			: Learning Rate starting value
			decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
			decay_step			: Step to apply decay
			dataset				: Dataset (class DataGenerator)
			training			: (bool) True for training / False for prediction
			w_summary			: (bool) True/False for summary of weight (to visualize in Tensorboard) (set true)
			w_loss				: (bool) used to weighted loss (didn't calculate loss on unvisible joints)
			tiny				: (bool) Activate Tiny Hourglass
			name				: name of the model
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
				# Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB) in NHWC format
				# NOTICE Set 256 unchanged
				self.img = tf.placeholder(dtype= tf.float32, shape= (None, 256, 256, 3), name = 'input_img')
				if self.w_loss:
					self.weights = tf.placeholder(dtype = tf.float32, shape = (None, self.outDim))
				# Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
				# Intermediate supervision: so need multiple by nStack = 4				
				self.gtMaps = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 64, 64, self.outDim))
			inputTime = time.time()
			print('-- Model Inputs : Done (' + str(int(abs(inputTime-startTime))) + ' sec.)')

			# Shape HG output: batchSize x nStack x 64 x 64 x outDim
			# Generate the graph of the whole hourglass model here.
			self.output = self._graph_hourglass(self.img)
			graphTime = time.time()
			print('-- Model Graph : Done (' + str(int(abs(graphTime-inputTime))) + ' sec.)')

			with tf.name_scope('loss'):
				# use sigmoid_cross_Ent to measure the loss
				if self.w_loss:
					self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')
				else:
					self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
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
		
		# use merge_all to (use only sess.run for one time and apply ops on all in the train collection)
		self.train_op = tf.summary.merge_all('train')
		self.test_op = tf.summary.merge_all('test') # test_summary if for validation
		self.weight_op = tf.summary.merge_all('weight') # used for conv function
		endTime = time.time()
		print('>>>>> Model created in (' + str(int(abs(endTime-startTime))) + ' sec.)')
		# every time is an object and if we don't need them then remove them
		del endTime, startTime, initTime, optimTime, minimTime, lrTime, accurTime, lossTime, graphTime, inputTime
		
	"""
	# --------------- Model Training --------------
	"""
	
	def weighted_bce_loss(self):
		""" Create Weighted Loss Function
		Don't calculate loss on unlabel joint (which unvisibility and has a blank heatmap)
		"""

		'''
		self.bceloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
		e1 = tf.expand_dims(self.weights, axis = 1, name = 'expdim01')
		e2 = tf.expand_dims(e1, axis = 1, name = 'expdim02')
		e3 = tf.expand_dims(e2, axis = 1, name = 'expdim03')
		return tf.multiply(e3, self.bceloss, name = 'lossW')
		'''
		loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps)
		weights = tf.reduce_sum(self.gtMaps, axis=[2, 3], keepdims=True)
		return weights * loss

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
					img_train, gt_train, weight_train = next(self.generator)
					if i % saveStep == saveStep - 1:
						if self.w_loss:
							_, cur_loss, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], 
								feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
						else:
							_, cur_loss, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], 
								feed_dict = {self.img : img_train, self.gtMaps: gt_train})
						# Save summary (Loss + Accuracy)
						# FileWriter to logdir_train
						self.train_summary.add_summary(summary, epoch*epochSize + i)
						self.train_summary.flush()
					else:
						if self.w_loss:
							_, cur_loss, = self.Session.run([self.train_rmsprop, self.loss], 
								feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
						else:
							_, cur_loss, = self.Session.run([self.train_rmsprop, self.loss], 
								feed_dict = {self.img : img_train, self.gtMaps: gt_train})
					cost += cur_loss
					avg_cost += cur_loss/epochSize
				epochfinishTime = time.time()
				
				# Save Weight (axis = epoch) for all the conv
				if self.w_loss:
					weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
				else :
					weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train})
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
				
				# Validation Part
				'''
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
			print('-- Final Loss: ' + str(cost) + '\n' + ' Loss Discimination: ' + str(100*self.resume['loss'][-1]/(self.resume['loss'][0] + 0.1)) + '%' )
			# print('-- Relative Accurancy Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) +'%')
			print('-- Training Time: ' + str(datetime.timedelta(seconds=time.time() - startTime)))
			
	def training_init(self, nEpochs = 100, epochSize = 100, saveStep = 20, valid_iter = 10, pre_trained = None):
		""" Initialize the training process

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
		""" Create Summary and Saver
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
		""" Create the `Network Graph`

			Args:
				inputs : TF Tensor (placeholder) of shape (None, 3, 256, 256) (The size of self.img)
		"""
		with tf.name_scope('model'):
			# preprocess the image
			with tf.name_scope('preprocessing'):
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
				with tf.name_scope('stacks'):
					for i in range(0, self.nStack):
						with tf.name_scope('stage_' + str(i)):
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

			else:
				# Full 4-Rank Houglass network 
				with tf.name_scope('stacks'):
					for i in range(0, self.nStack):
						with tf.name_scope('stage_' + str(i)):
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
			filters			: Number of filters (channels)
			kernel_size		: Size of kernel
			strides			: Stride
			name			: Name of the block
		Returns:
			conv			: Output Tensor (Convolved Input)
		"""
		with tf.name_scope(name):
			# Kernel for convolution, Xavier Initialisation
			# initialize the kernel weight using xavier method
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[1], filters]), name= 'weights')
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
		with tf.name_scope(name):
			# initialize the kernel weight using xavier method
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[1], filters]), name= 'weights')
			conv = tf.nn.conv2d(inputs, kernel, [1,1,strides,strides], padding='VALID', data_format='NCHW')
			norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
			if self.w_summary:
				with tf.device('/cpu:0'):
					# Adding a histogram summary makes it possible to visualize your data's distribution in TensorBoard
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return norm
	
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
			with tf.name_scope(name):
				norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
				pad = tf.pad(norm, np.array([[0,0],[0,0],[1,1],[1,1]]), name= 'pad')
				conv = self._conv(pad, int(numOut), kernel_size=3, strides=1, name= 'conv')
				return conv
		else:
			with tf.name_scope(name):
				# Standard convolution in the paper with kernel size [1 -> 3 -> 1]
				with tf.name_scope('norm_1'):
					norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
					conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, name= 'conv')
				with tf.name_scope('norm_2'):
					norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
					pad = tf.pad(norm_2, np.array([[0,0],[0,0],[1,1],[1,1]]), name= 'pad')
					conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, name= 'conv')
				with tf.name_scope('norm_3'):
					norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
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
		with tf.name_scope(name):
			if inputs.get_shape().as_list()[1] == numOut:
				return inputs
			else:
				conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
				return conv				
	
	def _residual(self, inputs, numOut, name = 'residual_block'):
		""" `Residual Unit`
		
		Args:
			inputs	: Input Tensor
			numOut	: Number of Output Features (channels)
			name	: Name of the block
			`Add on the conv2d and skip layer`
		"""
		with tf.name_scope(name):
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
		with tf.name_scope(name):
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
		"""
		self.joint_accur = []
		for i in range(len(self.joints)):
			# What's saved in self.nstack-1 is the last output of the network
			self.joint_accur.append(self._accur(self.output[:, self.nStack - 1, :, :,i], self.gtMaps[:, self.nStack - 1, :, :, i], self.batchSize))
	
	def _accur(self, pred, gtMap, num_image):
		""" Given a Prediction batch (pred) and a Ground Truth batch (gtMaps)
		for a single joint in each image of the batch(average in these images)
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
			err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
			# err is the distance-> bigger err means lower accurancy-> we need to 1-err
		return tf.subtract(tf.to_float(1), err/num_image)

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