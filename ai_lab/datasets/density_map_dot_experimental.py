import numpy as np
import cv2
import math

from ai_lab import storagemanager as sm


'''
	DensityMapDot(class)

	Description:
		DensityMapDot reads dataset information from a index.txt
		contained in the dir_path supplied. The Information 
		contains, data/label paths(relative to the directory)
		and the standard size (std_size) to whitch the data/label
		images (of various sizes) are scaled.
		The Images are loaded, divided by 256 and converted into 
		float64, so all values are between (0.0->(1.0-1/256)).
		Then gaussian kernels are added to each dot and the
		dot_count is kept. Arguments (args) can be passed along 
		like gauss_kernel_[radius/peak].

		[Only Greyscale images so image shape = [height,width,1] ]

	__init__(dir_path,args[dict]): 
		dir_path containing index.txt, indexing all 
		data and label samples in the dataset (train/test-set
		seperation occurs dynamically at runtime).

		args can contain the gauss_kernel_size, gauss_kernel_peak,
		validation_percentage, if none or certain parameters don't 
		exists use the standards.


	index.txt:
		headers
		|_data_image_path
		|_label_image_path
		|_std_size (standard size of the images)
'''
class DensityMapDot(object):

	'''
		DensityMapDot - Sample(class)

		Description:
			Variable collection for each data sample.
			Holds pre loaded data/label images for efficiency.

		__init__(data_file_path, label_file_path, data_image[scaled,gaussian], 
				label_image[scaled,gaussian], dot_count)
	'''
	class Sample(object):
		def __init__(self,data_file_path, label_file_path, 
			data_image, label_image,dot_count,position_matrix):

			self.data_file_path = data_file_path
			self.label_file_path = label_file_path
			self.data_image = data_image
			self.label_image = label_image
			self.dot_count = dot_count
			self.position_matrix = position_matrix


	hyper_param_dict = {
	"gauss_kernel_radius":10.0,
	"gauss_kernel_peak":1.0,
	"validation_percentage":20,
	"zoom_min":0.75,
	"zoom_max": 2.0,
	"std_size":1024
	}
	hyper_params = None

	dataset_loaded = False		# False => NOT usable; True => usable;
	index_file_path = None
	train_samples = None 		# Sample(class)[]
	validation_samples = None 	# Sample(class)[]

	compatibilities = ["cell_counter_density_map"]

	def __init__(self,dir_path,args={}):
		self.dir_path = dir_path

		self.set_hyper_params(args)
		self.dataset_loaded = self.load_dataset(dir_path)


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
	def load_dataset(self,dir_path):

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

			y_offset = c_y - height
			x_offset = c_x - width
			img_width, img_height=self.get_dims(img)

			img_x = max(0,x_offset)
			img_y = max(0,y_offset)
			gaus_x = max(-1*x_offset,0)
			gaus_y = max(-1*y_offset,0)
			gaus_x_s = max(0,(width*2)+min(0,img_width-(c_x + width)))
			gaus_y_s = max(0,(height*2)+min(0,img_height-(c_y + height)))

			img[img_y:y_offset+height*2, img_x:x_offset+width*2, 0] = comp_gaus[gaus_y:gaus_y_s,gaus_x:gaus_x_s,0] + img[img_y:y_offset+height*2, img_x:x_offset+width*2, 0]
			# * (1.0 - comp_gaus[:,:,0]/255.0)

			return img

		# Ret.shape = [x,2 => (x,y) <= normalized (0->1)]
		def create_position_matrix(label,width,height):
			y,x,_ = np.where(label > 0)
			position_matrix = np.asarray([ [float(x[i])/float(width),float(y[i])/float(height)] for i in range(len(y))]).astype(np.float64)#np.zeros([len(y),2],dtype=np.float64)
			return position_matrix


		index_file_path = '/'.join([dir_path,'index.txt'])
		# Index doesn't exist
		if sm.file_exists(index_file_path) == False:
			return False
		self.index_file_path = index_file_path

		### Read index.txt (csv format)
		with open(index_file_path,"r") as index_file:
			index_raw = index_file.read()
			index_line_split = index_raw.replace('\r','').split('\n')

			# Only/No headers
			if len(index_line_split) == 1:
				return False

			### Read dataset headers
			index_header_names = index_line_split[0].split(',')
			header_indices = {"data_name":None,"label_name":None,"std_size":None}

			# Note the indices for the header_names
			for i in range(len(index_header_names)):
				name = index_header_names[i]
				if name not in header_indices: continue
				header_indices[name] = i

			std_size = None
			double_std_size = None
			samples = []

			### Loading dataset samples
			for line in index_line_split[1:]:
				if line == "": continue

				# index.txt line 
				line_split = line.split(',')
				data_file_path = '/'.join([dir_path,line_split[header_indices["data_name"]]])
				label_file_path = '/'.join([dir_path,line_split[header_indices["label_name"]]])
				if std_size == None:
					std_size = int(line_split[header_indices["std_size"]])
				double_std_size = std_size*2
				hp["std_size"] = std_size

				# Image doesn't exist
				if sm.file_exists(data_file_path) == False or sm.file_exists(label_file_path) == False:
					continue

				### Loading Data/Label Image
				# load
				data_image = cv2.imread(data_file_path,0)	# Always Greyscale
				label_image = cv2.imread(label_file_path,0)	# Always Greyscale

				# get_dims
				data_width,data_height = self.get_dims(data_image)
				label_width,label_height = self.get_dims(label_image)

				# reshape
				data_image = data_image.reshape((data_height,data_width,1))
				label_image = label_image.reshape((label_height,label_width,1))
				
				# Create position matrix
				position_matrix = create_position_matrix(label_image,label_width,label_height)


				# pixel/256
				data_image = np.divide(data_image, 256.0).astype(np.float64)
				label_image = np.divide(label_image, 256.0).astype(np.float64)


				### Resize Images
				data_image = cv2.resize(data_image,(double_std_size, double_std_size),interpolation=cv2.INTER_CUBIC)
				label_image = cv2.resize(label_image,(double_std_size, double_std_size),interpolation=cv2.INTER_CUBIC)
				
				# reshape again
				data_image = data_image.reshape((double_std_size,double_std_size,1))
				label_image = label_image.reshape((double_std_size,double_std_size,1))

				samples.append(self.Sample(data_file_path,label_file_path,data_image,label_image,-1,position_matrix))

				'''
				# count dot's
				y,x,_ = np.where(label_image > 0.5)
				dot_count = len(y)

				# Apply Gauss-Kernel
				radius = int(hp["gauss_kernel_radius"] * (float(label_width)/float(std_size)))		# Cope with different image sizes
				peak = hp["gauss_kernel_peak"]

				density_map = np.zeros(label_image.shape,dtype=np.float64)
				for i in range(len(y)):
					density_map = draw_gaussian(radius, peak, density_map, x[i], y[i])

				label_image = density_map

				# reshape again :(
				data_image = data_image.reshape((std_size,std_size,1))
				label_image = label_image.reshape((std_size,std_size,1))

				### Add Sample
				samples.append(self.Sample(data_file_path,label_file_path,data_image,label_image,dot_count))
				''' # Gauss-DensityMap stuff


			### Divide samples into train/validation samples
			sample_count = len(samples)
			train_sample_count = int((float(sample_count) / 100.0)*(100.0-float(hp["validation_percentage"])))
			validation_sample_count = sample_count-train_sample_count

			
			r = np.arange(0,sample_count,dtype=np.int32)
			np.random.shuffle(r)

			train_samples = [samples[r[x]] for x in range(0,train_sample_count)]
			validation_samples = [samples[r[x]] for x in range(train_sample_count,sample_count)]

			self.train_samples = train_samples
			self.validation_samples = validation_samples
			return True


	'''
		Menditory Dataset Functions
		|_ get_sample_count(val=T/F)
		|_ get_sample_range(range,val=T/F)
		|_ get_sample(idx,val=T/F)
		|_ get_batch(batch_size,val=T/F)
		|_ get_sample_info(idx,val=T/F)
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