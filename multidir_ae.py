import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
from PIL import Image
from glob import glob
import os
from helper_experimental import *
import ai_lab.evaluator as evaluator
import ai_lab.transform.trans2d as trans2d
import cv2

#import models
import models_small as models
import utils

import argparse

parser = argparse.ArgumentParser(description='')
#parser.add_argument('--test',dest='test',default='false',type=str,help='test on validation set.')
#parser.add_argument('--model_name',dest='model_name',default='model_0_x_test_ckpt',type=str,help='file name of model.')
parser.add_argument('--run_idx', dest='run_idx',default=0,type=int,help='index of training run.')
parser.add_argument('--dataset',dest='dataset',default='./dataset/',type=str,help='path of dataset folder.')
#parser.add_argument('--label_name',dest='label_name',default='gauss_v3',type=str,help='string before label idx.')
args = parser.parse_args()


# Size of image frames
height = 128
width = 128
color_channels=3
output_color_channels = 3


# This trains the GAN on MNIST
def train(args):
	init_dataset(args.dataset)
	
	batch_size = 16
	samples_per_patch = 1
	iterations = 5000000
	itr_index = args.run_idx
	sample_directory = './saves/'+str(itr_index)+'/figs'	# Directory to save sample images from the generator in.
	model_directory = './saves/'+str(itr_index)+'/models'	# Directory to save trained model to.
	load_model = True	# Whether to load the model or begin training from scratch.
	pretrain_itr = 0

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()


	if not os.path.isdir('./saves'):
		os.mkdir('./saves')
	if not os.path.isdir('./saves/'+str(itr_index)):
		os.mkdir('./saves/'+str(itr_index))

	with tf.Session() as sess:
		merged = tf.summary.merge_all()
		if not os.path.isdir('./saves/'+str(itr_index)+'/logs'):
			os.mkdir('./saves/'+str(itr_index)+'/logs')
		train_writer = tf.summary.FileWriter('./saves/'+str(itr_index)+'/logs/'+str(itr_index)+'_1',sess.graph)

		sess.run(init)

		if load_model == True:
			print '[ Model Loading... ]'
			ckpt = tf.train.get_checkpoint_state(model_directory)
			if ckpt is not None:# and os.path.isdir(ckpt.model_checkpoint_path):
				saver.restore(sess,ckpt.model_checkpoint_path)
				print '[ Model Loaded! ]'
			else:
				print '[ !Model could NOT be loaded! ]'
		
		for i in range(1,iterations+1):
			images_x,images_y = get_training_batch(batch_size,size=height,channels=color_channels,color_jitter=False)

			ys = (np.reshape(images_y[:,:,:,:output_color_channels],[batch_size*samples_per_patch,height,width,output_color_channels])-0.5)*2.0	# Transform to be between -1 and 1
			xs = (np.reshape(images_x,[batch_size*samples_per_patch,height,width,color_channels])-0.5)*2.0

			
			_ = sess.run(optim_op,feed_dict={
				x_in: xs,
				y_in: ys,
				r_in: np.random.random_sample(np.array([xs.shape[0]]+list(latent_shape[1:])))/10.0 - 0.05
				})

			if i % 10 == 0:
				summary = sess.run(merged,feed_dict={
				x_in: xs,
				y_in: ys,
				r_in: np.random.random_sample(np.array([xs.shape[0]]+list(latent_shape[1:])))/10.0 - 0.05
				})
				train_writer.add_summary(summary,i)

			if i % 200 == 0:
				print "[Iter]",i
				val_batch_size = 3
				images_x,images_y = get_validation_batch(val_batch_size,size=height,channels=color_channels)
				xs = (np.reshape(images_x,[val_batch_size,height,width,color_channels])-0.5) * 2.0
				ys = (np.reshape(images_y[:,:,:,:output_color_channels],[val_batch_size,height,width,output_color_channels])-0.5) * 2.0
				
				x_x,xrx,yrx, y_y,x_y,xry = sess.run(validation_dec_range, feed_dict={x_in: xs,
				y_in: ys,
				r_in: np.random.random_sample(np.array([xs.shape[0]]+list(latent_shape[1:])))/10.0 - 0.05
				})

				s_image_parts = []
				for j in range(val_batch_size):
					s_part_0 = np.concatenate([xs[j],x_x[j],xrx[j],yrx[j]],axis=0)
					s_part_1 = np.concatenate([ys[j],y_y[j],x_y[j],xry[j]],axis=0)
					s_image_parts.append(np.concatenate([s_part_0,s_part_1],axis=1))
					#print s_part_0.shape,s_part_1.shape,s_image_parts[-1].shape

				s_images = np.concatenate([s_image_parts[j] for j in range(val_batch_size)],axis=1)
				#print s_images.shape

				if not os.path.exists(sample_directory):
					os.makedirs(sample_directory)
				
				save_images(np.array([s_images]),[1,1],sample_directory+'/test_fig'+str(i)+'.png')
				#print "Max-Val:",np.amax(sample_G)

			if i % 5000 == 0 and i != 0:
				if not os.path.exists(model_directory):
					os.makedirs(model_directory)
				saver.save(sess,model_directory+'/model_'+str(itr_index)+'_fluorescence_elsa-'+str(i)+"ckpt")
				print "Saved Model"


# Connecting it all together
tf.reset_default_graph()

latent_shape = models.get_latentspace_shape(width,height)
print "latent_shape:",latent_shape

# IN Placeholders
x_in = tf.placeholder(shape=[None,height,width,color_channels],dtype=tf.float32)
y_in = tf.placeholder(shape=[None,height,width,output_color_channels],dtype=tf.float32)
r_in = tf.placeholder(shape=latent_shape,dtype=tf.float32)

# ENCODERS + LATENTSPACES
enc_x = models.encX(x_in,width,height)
enc_y = models.encY(y_in,width,height)
L_X = enc_x
LrX = enc_x+r_in
L_Y = enc_y
LrY = enc_y+r_in

# DECODERS
dec_x_x = models.decX(L_X,output_color_channels)
dec_xrx = models.decX(LrX,output_color_channels)
dec_yrx = models.decX(LrY,output_color_channels)
dec_y_y = models.decY(L_Y,output_color_channels)
dec_x_y = models.decY(L_X,output_color_channels)
dec_xry = models.decY(LrX,output_color_channels)

# yry ???
validation_dec_range = [dec_x_x, dec_xrx, dec_yrx, \
						dec_y_y, dec_x_y, dec_xry]


# LOSSES
loss_dec_x_x = utils.loss_mse(dec_x_x,x_in,'loss_dec_x_x')
loss_dec_xrx = utils.loss_mse(dec_xrx,x_in,'loss_dec_xrx')
loss_dec_yrx = utils.loss_mse(dec_yrx,x_in,'loss_dec_yrx')
loss_dec_y_y = utils.loss_mse(dec_y_y,y_in,'loss_dec_y_y')
loss_dec_x_y = utils.loss_mse(dec_x_y,y_in,'loss_dec_x_y')
loss_dec_xry = utils.loss_mse(dec_xry,y_in,'loss_dec_xry')
loss_L_X_L_Y = utils.loss_mse(L_X,L_Y,'loss_L_X_L_Y')

# UNIVERSAL TRAINER
trainer = tf.train.AdamOptimizer(learning_rate=0.0001,beta1=0.5)


# GRADIENTS
grads_dec_x_x = trainer.compute_gradients(loss_dec_x_x)
grads_dec_xrx = trainer.compute_gradients(loss_dec_xrx)
grads_dec_yrx = trainer.compute_gradients(loss_dec_yrx)
grads_dec_y_y = trainer.compute_gradients(loss_dec_y_y)
grads_dec_x_y = trainer.compute_gradients(loss_dec_x_y)
grads_dec_xry = trainer.compute_gradients(loss_dec_xry)
grads_L_X_L_Y = trainer.compute_gradients(loss_L_X_L_Y)


# OPTIMIZERS
optim_dec_x_x = trainer.apply_gradients(grads_dec_x_x)
optim_dec_xrx = trainer.apply_gradients(grads_dec_xrx)
optim_dec_yrx = trainer.apply_gradients(grads_dec_yrx)
optim_dec_y_y = trainer.apply_gradients(grads_dec_y_y)
optim_dec_x_y = trainer.apply_gradients(grads_dec_x_y)
optim_dec_xry = trainer.apply_gradients(grads_dec_xry)
optim_L_X_L_Y = trainer.apply_gradients(grads_L_X_L_Y)
optim_op = [optim_dec_x_x, optim_dec_xrx, optim_dec_yrx,\
			optim_dec_y_y, optim_dec_x_y, optim_dec_xry,\
			optim_L_X_L_Y]

train(args)