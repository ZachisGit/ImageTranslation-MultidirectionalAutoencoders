import cv2
import numpy as np
import math

''' Get_X_Region - Functions '''

'''
	get_rotated_region(data[],labels[],region_size, region_count_per_sample)

	Description:
		Takes multiple samples, the size of the regions to be cut and
		the amount of regions to be extracted from each sample. 
		Then rotate each sample randomly and cut regions from the 
		centered circle with radius sample_size/2 (~21% loss compared to
		rotation specific discrimination method)
		[Works only with squares (a*b => b=a)]

	Returns:
		Regions for each image in order of samples
		ret data_regions,label_regions
'''
def get_rotated_region(data, labels, region_size, region_count_per_sample):

	# Returns a random xy pos in a circle with center-offset
	# of o_x,o_y and a radius of max_r
	def get_random_circular_xy(o_x, o_y,max_r):
		r = np.random.uniform(0,max_r)
		angle = np.random.uniform(0,math.pi*2.0)

		x=o_x+r*math.sin(angle)
		y=o_y+r*math.cos(angle)
		return x,y

	width,height, channels = get_dims_hwc(data[0])
	sample_count = len(data)

	# Circular transform variables
	region_r = math.sqrt(2.0*math.pow(region_size,2))/2.0	# radius of the circle containing the region_size rectangle
	max_r = width / 2.0 - region_r 	# max radius for random radius selection

	data_regions = np.zeros((sample_count*region_count_per_sample,region_size,region_size,channels))
	label_regions = np.zeros((sample_count*region_count_per_sample,region_size,region_size,1))

	for s in range(sample_count):
		# Rotate sample randomly
		matrix = cv2.getRotationMatrix2D((width/2,height/2),np.random.randint(0,360),1)
		data_r = cv2.warpAffine(data[s],matrix,(width,height))
		label_r = cv2.warpAffine(labels[s],matrix,(width,height))

		# reshape
		data_r = data_r.reshape((height,width,1))
		label_r = label_r.reshape((height,width,1))

		for r in range(region_count_per_sample):
			# Calculate start xy of region
			circle_x, circle_y = get_random_circular_xy(width/2,height/2,max_r)
			start_x = int(circle_x - region_size/2)	# Start offset of region
			start_y = int(circle_y - region_size/2)  # x,y

			# Cut out region
			data_region = data_r[start_y:start_y+region_size, 
			start_x:start_x+region_size] 
			label_region = label_r[start_y:start_y+region_size, 
			start_x:start_x+region_size]

			idx = s*region_count_per_sample+r
			data_regions[idx] = data_region
			label_regions[idx] = label_region

	return data_regions,label_regions


def get_all_regions(data,labels,region_size,use_labels=True):
	width,height,channels = get_dims_hwc(data[0])
	data = np.reshape(data,[len(data),height,width,channels])

	sample_rows = int(math.floor(float(height)/float(region_size)))
	sample_cols = int(math.floor(float(width)/float(region_size)))
	regions_per_sample = sample_rows*sample_cols

	data_r = np.zeros((regions_per_sample*len(data),region_size,region_size,channels),dtype=np.float64)
	if use_labels == False:
		label_r = None
	else:
		label_r = np.zeros((regions_per_sample*len(labels),region_size,region_size,1),dtype=np.float64)

	for i in range(len(data)):
		for y in range(sample_rows):
			for x in range(sample_cols):
				c_data_r = data[i,y*region_size:y*region_size+region_size, x*region_size:x*region_size+region_size]
				if use_labels == True:
					c_label_r = labels[i,y*region_size:y*region_size+region_size, x*region_size:x*region_size+region_size]

				idx = i*regions_per_sample+y*sample_cols+x
				data_r[idx] = c_data_r
				if use_labels == True:
					label_r[idx] = c_label_r
	return data_r,label_r



''' Augmentations '''

def zca_whitening(data):
	# Flatten data
	data_1d,shape = vector_array_n_to_1d(data)
	
	for i in range(shape[0]):

		inputs = data_1d[i].reshape((1,len(data_1d[i])))

		# Compute zca_matrix
		sigma = np.dot(inputs,inputs.T)/inputs.shape[1]
		U,S,V = np.linalg.svd(sigma)
		epsilon = 0.1
		zca_matrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)
		
		# Apply zca_matrix to input vectors
		res = np.dot(zca_matrix,inputs)
		data_1d[i] = res

	# Reshape input_vectors
	return vector_array_1d_to_n(data_1d,shape)

def color_jitter(data,min_value=0.0,max_value=0.2):
	jitter = np.random.uniform(min_value,max_value,size=get_shape_size(data))
	noise = jitter.reshape(data.shape)
	res = np.add(data,noise)
	return res


''' Helpers '''

def show_image_hwc(image,win_name="image",width=600,height=600,wait_key=True, destroy_window=True):	
	cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
	cv2.resizeWindow(win_name, width,height)
	cv2.imshow(win_name,image)

	if wait_key: cv2.waitKey(0)
	if destroy_window: cv2.destroyAllWindows()

def get_dims_hwc(image):
	width = image.shape[1]
	height = image.shape[0]
	if len(image.shape) < 3:
		channel = 1
	else:
		channel = image.shape[2]
	return width,height,channel

# Returns the multiplied size of the shape
def get_shape_size(array):
	size = 1
	for i in range(len(array.shape)):
		size*=array.shape[i]
	return size

# input numpy array
# Ret:	1d_vector, original_shape
def vector_array_n_to_1d(vector):
	ori_shape = vector.shape
	flat_vec = vector.flatten(0)
	flat_vec = flat_vec.reshape(ori_shape[0],len(flat_vec)/ori_shape[0])
	return flat_vec, ori_shape

# input numpy array, shape
# Ret:	nd_vector with shape shape
def vector_array_1d_to_n(vector, shape):
	ori = vector.reshape(shape)
	return ori