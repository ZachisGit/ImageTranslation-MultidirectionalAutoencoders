import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from utils import *

enc_x_reuse = False
enc_y_reuse = False
dec_x_reuse = False
dec_y_reuse = False

enc_channels = [64,128,64,16]
dec_channels = [64,128,128,64]

# Global Variables
def get_latentspace_shape(w,h):
	c_w = int(w/np.power(2,4))
	c_h = int(h/np.power(2,4))
	return np.array([None,c_h,c_w,enc_channels[-1]])

### ENCODERS ###
def encX(x,w,h):
	global enc_x_reuse

	kernel_sizes = [4,4,4,4]
	sample_rates = [2,2,2,2]

	model = x
	c_w,c_h = w,h

	with tf.variable_scope('encX',reuse=enc_x_reuse):
		# Adjust reuse param
		if enc_x_reuse == False:
			enc_x_reuse = True

		for i in range(len(kernel_sizes)):
			c_w /= sample_rates[i]
			c_h /= sample_rates[i]
			model = downsample(model,enc_channels[i],kernel_sizes[i],sample_rates[i],'encX_downsample-ksize_'+str(kernel_sizes[i])+'-wh_'+str(c_w)+'x'+str(c_h),activation=lrelu)
			print "enc_channels:",model.shape

	return model

def encY(y,w,h):
	global enc_y_reuse

	kernel_sizes = [4,4,4,4]
	sample_rates = [2,2,2,2]

	model = y
	c_w,c_h = w,h

	with tf.variable_scope('encY',reuse=enc_y_reuse):
		# Adjust reuse param
		if enc_y_reuse == False:
			enc_y_reuse = True

		for i in range(len(kernel_sizes)):
			c_w /= sample_rates[i]
			c_h /= sample_rates[i]
			model = downsample(model,enc_channels[i],kernel_sizes[i],sample_rates[i],'encY_downsample-ksize_'+str(kernel_sizes[i])+'-wh_'+str(c_w)+'x'+str(c_h),activation=lrelu)

	return model



### DECODERS ###
def decX(L,out_channels):
	global dec_x_reuse

	kernel_sizes = [4,4,4,1]
	sample_rates = [2,2,2,2]

	model = L
	c_h,c_w = L.shape[1:3]

	with tf.variable_scope('decX',reuse=dec_x_reuse):
		# Adjust reuse param
		if dec_x_reuse == False:
			dec_x_reuse = True

		for i in range(len(kernel_sizes)):
			c_w *= sample_rates[i]
			c_h *= sample_rates[i]
			model = upsample(model,dec_channels[i],kernel_sizes[i],sample_rates[i],'decX_upsample-ksize_'+str(kernel_sizes[i])+'-wh_'+str(c_w)+'x'+str(c_h),activation=lrelu)
			print "dec_channels:",model.shape
		model = conv2d(model,out_channels,1,'decX_Out-wh_'+str(c_w)+'x'+str(c_h)+'-actifn_tanh',activation=tf.nn.tanh)

	return model

def decY(L,out_channels):
	global dec_y_reuse

	kernel_sizes = [4,4,4,1]
	sample_rates = [2,2,2,2]

	model = L
	c_h,c_w = L.shape[1:3]

	with tf.variable_scope('decY',reuse=dec_y_reuse):
		# Adjust reuse param
		if dec_y_reuse == False:
			dec_y_reuse = True

		for i in range(len(kernel_sizes)):
			c_w *= sample_rates[i]
			c_h *= sample_rates[i]
			model = upsample(model,dec_channels[i],kernel_sizes[i],sample_rates[i],'decY_upsample-ksize_'+str(kernel_sizes[i])+'-wh_'+str(c_w)+'x'+str(c_h),activation=lrelu)

		model = conv2d(model,out_channels,1,'decY_Out-wh_'+str(c_w)+'x'+str(c_h)+'-actifn_tanh',activation=tf.nn.tanh)
	return model


