import numpy as np
import cv2
import math
from glob import glob
import os
import re
from ai_lab import storagemanager as sm

class ImageFolder(object):

	class Sample(object):
		def __init__(self,data_file_path, label_file_path):

			self.data_file_path = data_file_path
			self.label_file_path = label_file_path
			self.data_image = None
			self.label_image = None

		def get_image(self,hp):
			if self.data_file_path is None or \
			   self.label_file_path is None:
			   return None

			if self.data_image is not None and \
			   self.label_image is not None:

				back = [self.data_image,self.label_image]
				if hp['keep_mem'] == False:
					self.data_image = None
					self.label_image = None
				return back
	

			assert os.path.isfile(self.data_file_path)
			assert os.path.isfile(self.label_file_path)

			data = cv2.imread(self.data_file_path, cv2.IMREAD_COLOR if hp['rgb'] else cv2.IMREAD_GRAYSCALE)
			label = cv2.imread(self.label_file_path,cv2.IMREAD_COLOR if hp['rgb'] else cv2.IMREAD_GRAYSCALE)

			if hp['keep_mem'] == True:
				self.data_image = data
				self.label_image = label

			return data,label

	hyper_param_dict = {
	"zoom_min":0.75,
	"zoom_max": 2.0,
	"std_size": 128,
	"keep_mem":False,
	"rgb": True
	}

	hyper_params = None

	dataset_loaded = False		# False => NOT usable; True => usable;
	folder_path = None
	train_samples = None 		# Sample(class)[]
	validation_samples = None 	# Sample(class)[]

	compatibilities = ["image_folder"]

	def __init__(self,folder_path,args={}):
		self.folder_path = folder_path

		self.set_hyper_params(args)
		self.dataset_loaded = self.load_dataset(folder_path)


	'''
		Set the hyper-parameters based on the hyper_param_dict
		if the dict is None raise NotImplementedError
		just igonre hyper_parameters not set in hyper_param_dict
		
		class-vars:	hyper_params, hyper_param_dict
		Ret: -
	'''
	def set_hyper_params(self, hyper_params):
		if self.hyper_param_dict == None:
			raise NotImplementedError()

		# Set all parameters defined in hyper_params,
		# for the rest use the predefined values in
		# self.hyper_param_dict.
		n_hyper_params = {}
		for key,value in self.hyper_param_dict.iteritems():
			if key in hyper_params:
				n_hyper_params[key] = hyper_params[key]
			else:
				n_hyper_params[key] = value

		self.hyper_params = n_hyper_params


	# Ret: Is model compatible with evaluator (boolean)
	# class-vars:	compatibilities
	def check_compatibility(self,model):
		model_comp = model.model_compatiblity
		return model_comp in self.compatibilities

	# width, height
	def get_dims(self,image):
		return image.shape[1],image.shape[0]

	# True/False loading the dataset worked
	# class-vars:	index_file_path, train_samples, validation_samples
	def load_dataset(self,folder_path):

		hp = self.hyper_params

		train_data_dir = folder_path+'/trainA/'
		train_label_dir = folder_path+'/trainB/'
		test_data_dir = folder_path+'/testA/'
		test_label_dir = folder_path+'/testB/'
		no_test_data = False

		if not os.path.isdir(train_data_dir) or not os.path.isdir(train_label_dir):
			return False
		elif not os.path.isdir(test_data_dir) or not os.path.isdir(test_label_dir):
			no_test_data = True

		def load_data_label_pairs(data_folder,label_folder):
			extensions = ['jpg','jpeg','png','bmp']
			data_files = []
			label_files = []
			samples = []

			for e in extensions:
				data_files.extend(glob(data_folder+'*.'+e))
				label_files.extend(glob(label_folder+'*.'+e))
	

			# extract label id's
			label_dict={}
			for i in range(len(label_files)):
				nums = re.findall(r'\d+',label_files[i])
				if len(nums) == 0:
					continue
				label_dict[nums[-1]] = i

			# extract data id's
			for i in range(len(data_files)):
				nums = re.findall(r'\d+',data_files[i])
				num = -1 if len(nums) == 0 else nums[-1]

				if num in label_dict.keys():
					samples.append(self.Sample(data_files[i],label_files[label_dict[num]]))

			return samples

		train_samples = load_data_label_pairs(train_data_dir,train_label_dir)
		if train_samples is None:
			return False
		self.train_samples = train_samples


		if not no_test_data:
			test_samples = load_data_label_pairs(test_data_dir,test_label_dir)
			if test_samples is not None:
				self.validation_samples = test_samples

		return True


	def get_batch(self,batch_size,validation=False):
		samples = self.validation_samples if validation else self.train_samples

		if samples is None or len(samples) == 0:
			return None

		r = np.arange(0,len(samples))
		np.random.shuffle(r)
		size = min(batch_size,len(samples))

		data_batch,label_batch = [],[]
		for i in range(size):
			data,label = samples[r[i]].get_image(self.hyper_params)
			data_batch.append(data)
			label_batch.append(label)

		return data_batch,label_batch

	'''
		Menditory Dataset Functions
		|_ get_sample_count(val=T/F)
		|_ get_sample_range(range,val=T/F)
		|_ get_sample(idx,val=T/F)
		|_ get_batch(batch_size,val=T/F)
		|_ get_sample_info(idx,val=T/F)
	'''
	'''
	# Takes one sample and returns the zoomed data,label
	# and replaces label by density_map (might not be neccessary)
	def _get_training_sample(self,data,label,position_matrix,zoom):
		hp = self.hyper_params

		def draw_gaussian(radius,peak, img,c_x, c_y):
			width,height = radius,radius
			radius /= 4.0  # ???

			gaus_grid = np.zeros((width,height,1),dtype=np.float64)
			gaus_center = width-1

			for y in range(width):
				for x in range(height):
					gaus_grid[y,x,0] = math.exp(-( (math.pow(x-gaus_center,2)+math.pow(y-gaus_center,2)) / 
						math.pow(2*(radius),2)))*peak

			left_gaus = np.vstack((gaus_grid,np.flipud(gaus_grid)))
			comp_gaus = np.hstack((left_gaus,np.fliplr(left_gaus)))

			y_offset = int(c_y) - height
			x_offset = int(c_x) - width
			img_width, img_height=self.get_dims(img)

			img_x = max(0,x_offset)
			img_y = max(0,y_offset)
			gaus_x = max(-1*x_offset,0)
			gaus_y = max(-1*y_offset,0)
			gaus_x_s = max(0,(width*2)+min(0,img_width-(int(c_x) + width)))
			gaus_y_s = max(0,(height*2)+min(0,img_height-(int(c_y) + height)))

			#print left_gaus.shape,comp_gaus.shape
			#print img_x,img_y,gaus_x,gaus_y,gaus_x_s,gaus_y_s,img_width,img_height,x_offset,y_offset

			img[img_y:y_offset+height*2, img_x:x_offset+width*2, 0] = comp_gaus[gaus_y:gaus_y_s,gaus_x:gaus_x_s,0] + img[img_y:y_offset+height*2, img_x:x_offset+width*2, 0]
			# * (1.0 - comp_gaus[:,:,0]/255.0)

			return img

		def position_matrix_to_coordinates(position_matrix,width,height):
			#print position_matrix.shape
			return np.multiply(position_matrix,np.asarray([width,height]))

		if zoom:
			zoom_factor = (np.random.uniform() * (hp["zoom_max"]-hp["zoom_min"]) + hp["zoom_min"]) / 2.0		
		else:
			zoom_factor = 0.5
		
		n_data = cv2.resize(data,None,fx=zoom_factor,fy=zoom_factor,interpolation=cv2.INTER_CUBIC)
		n_label = n_data.copy()*0.1#np.zeros_like(n_data)#cv2.resize(label,None,fx=zoom_factor,fy=zoom_factor,interpolation=cv2.INTER_CUBIC)
		n_data = n_data.reshape([n_data.shape[0],n_data.shape[1],1])
		n_label = n_label.reshape([n_label.shape[0],n_label.shape[1],1])

		if len(position_matrix) > 0:
			width,height = self.get_dims(n_label)
			coordinates = position_matrix_to_coordinates(position_matrix,width,height)
			std_size = hp["std_size"]
			radius = int(hp["gauss_kernel_radius"] * 1.0) #(float(width)/float(std_size)))		# Cope with different image sizes
			peak = hp["gauss_kernel_peak"]
			
			density_map = n_label
			for i in range(len(position_matrix)):
				density_map = draw_gaussian(radius,peak,density_map,coordinates[i,0],coordinates[i,1])
			n_label = density_map
			density_map = None

		return n_data,n_label




	# Returns train/validation samples array
	def _get_train_validation_samples(self,validation):
		if validation:
			return self.validation_samples
		else:
			return self.train_samples


	def get_sample_count(self,validation=False):
		assert self.dataset_loaded

		def _get_sample_count(samples):
			assert samples != None
			return len(samples)

		return _get_sample_count(self._get_train_validation_samples(validation))

	def get_sample_range(self,r,validation=False,zoom=True):
		assert self.dataset_loaded

		def _get_sample_range(samples,r):
			assert samples != None

			data_r, label_r,position_matricies_r = [],[],[]
			sample_count = len(samples)
			assert sample_count > 0
			
			for i in r:
				if i >= sample_count: continue

				# Calc DensityMap (Label) + Zoom
				data = samples[i].data_image
				label = samples[i].label_image
				position_matrix = samples[i].position_matrix

				data,label = self._get_training_sample(data,label,position_matrix,zoom)

				data_r.append(data)
				label_r.append(label)
				position_matricies_r.append(position_matrix)

			return data_r,label_r,position_matricies_r

		return _get_sample_range(self._get_train_validation_samples(validation),r)

	# Returns range + cell_count
	def get_sample_range_info(self,r,validation=False):
		assert self.dataset_loaded

		def _get_sample_range_info(samples,r):
			assert samples != None

			data, labels, dot_count = [],[],[]
			sample_count = len(samples)
			assert sample_count > 0

			for i in r:
				if i >= sample_count: continue
				data.append(samples[i].data_image)
				labels.append(samples[i].label_image)
				dot_count.append(samples[i].dot_count)

			return np.asarray(data),np.asarray(labels),np.asarray(dot_count)

		return _get_sample_range_info(self._get_train_validation_samples(validation),r)

	def get_sample(self,idx,validation=False,zoom=True):
		assert self.dataset_loaded

		def _get_sample(samples,idx):
			assert samples != None

			sample_count = len(samples)
			if idx >= sample_count:
				return None

			data = samples[idx].data_image
			label = samples[idx].label_image
			position_matrix = samples[idx].position_matrix

			# Calc DensityMap (Label) + Zoom
			data,label = self._get_training_sample(data,label,position_matrix,zoom)

			return data,label,position_matrix

		return _get_sample(self._get_train_validation_samples(validation),idx)

	def get_batch(self,batch_size,validation=False,zoom=True):
		assert self.dataset_loaded

		def _get_batch(samples,batch_size):
			assert samples != None

			sample_count = len(samples)
			assert sample_count > 0

			r = np.arange(0,max(sample_count,batch_size))
			np.random.shuffle(r)
			r_i = r[:batch_size]

			data_batch_shape = [batch_size]
			label_batch_shape = [batch_size]
			data_batch_shape.extend(samples[0].data_image.shape)
			label_batch_shape.extend(samples[0].label_image.shape)

			data_batch = []#np.zeros(data_batch_shape,dtype=np.float64)
			label_batch = []#np.zeros(label_batch_shape,dtype=np.float64)
			position_matricies =[]

			for i in range(batch_size):
				sample_idx = r_i[i]%sample_count

				data = samples[sample_idx].data_image
				label = samples[sample_idx].label_image
				position_matrix = samples[sample_idx].position_matrix

				# Calc DensityMap (Label) + Zoom
				data,label = self._get_training_sample(data,label,position_matrix,zoom)

				data_batch.append( data)
				label_batch.append(label)
				position_matricies.append(position_matrix)

			return data_batch,label_batch,position_matricies

		return _get_batch(self._get_train_validation_samples(validation),batch_size)

	def get_sample_info(self,idx,validation=False):
		assert self.dataset_loaded

		def _get_sample_info(samples,idx):
			assert samples != None

			sample_count = len(samples)
			if idx >= sample_count:
				return None,None,None

			sample = samples[i]
			return sample.data_image,sample.label_image,sample.dot_count

		return _get_sample_info(self._get_train_validation_samples(validation),idx)
	'''