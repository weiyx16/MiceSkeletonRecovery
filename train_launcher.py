"""
Deep Mice Pose Estimation Using Stacked Hourglass Network

Project by @Eason
Adapted from @Walid Benbihi [source code]github : https://github.com/wbenbihi/hourglasstensorlfow/

---
Main function
---
"""

import configparser
import argparse
import sys
import os
from hourglass_tiny import HourglassModel
from datagen import DataGenerator


def process_config(conf_file):
	"""
		Read the config file from the path of conf_file

		Including:  
			DataSetHG
			Network
			Train
			Validation
			Saver
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	# convert the parameters in each section to params
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Evaluation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


if __name__ == '__main__':
	# add some parameters from the terminal
	parser = argparse.ArgumentParser(description='Launch the training of the Hourglass model.', add_help=True, epilog='Just a test for this parameter')
	parser.add_argument('--version', action='version', version='Version 1.0')
	parser.add_argument('--cfg', required=False, default = './config.cfg', help='The path for your config file')
	args = parser.parse_args()

	print('>>>>> Parsing Config File From %s' %(args.cfg))
	params = process_config(args.cfg)
	
	print('>>>>> Creating Dataset Now')
	# dataset.train_set is the table of the training set's names
	dataset = DataGenerator(joints_name = params['joint_list'],img_dir = params['img_directory'], train_data_file = params['training_txt_file'],
							camera_extrinsic = params['camera_extrinsic'], camera_intrinsic = params['camera_intrinsic'])
	dataset._create_train_table()

	# nfeats:256, nstacks:4 nmodules:1(not used)
	# nlow:4 (Number of downsampling in one stack)
	# mcam:false (attention system(not needed))
	# name:pretrained model
	# tiny:false weighted_loss:false

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], 
		nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'], training=True, 
		drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], 
		dataset=dataset, name=params['name'], w_summary = True, logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= params['tiny'],
		w_loss=params['weighted_loss'] , joints= params['joint_list'], gpu_frac=params['gpu_frac'], model_save_dir=params['model_save_dir'])
	
	print('>>>>> Creating Hourglass Model')
	model.generate_model()
	model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],valid_iter=params['valid_iteration'], pre_trained = params['pretrained_model'])
