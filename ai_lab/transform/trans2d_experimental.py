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
def get_rotated_region(data, labels,position_matricies, region_size, region_count_per_sample):

	# Returns a random xy pos in a circle with center-offset
	# of o_x,o_y and a radius of max_r
	def get_random_circular_xy(o_x, o_y,max_r):
		r = np.random.uniform(0,max_r)
		angle = np.random.uniform(0,math.pi*2.0)

		x=o_x+r*math.sin(angle)
		y=o_y+r*math.cos(angle)
		return x,y

	sample_count = len(data)
	data_regions = np.zeros((sample_count*region_count_per_sample,region_size,region_size,1))
	label_regions = np.zeros((sample_count*region_count_per_sample,region_size,region_size,1))
	cell_counts = np.zeros((sample_count*region_count_per_sample),dtype=np.int32)

	for s in range(sample_count):

		width,height, channels = get_dims_hwc(data[s])

		# Circular transform variables
		region_r = math.sqrt(2.0*math.pow(region_size,2))/2.0	# radius of the circle containing the region_size rectangle
		max_r = width / 2.0 - region_r 	# max radius for random radius selection

		# Rotate sample randomly
		matrix = cv2.getRotationMatrix2D((width/2,height/2),np.random.randint(0,360),1)
		data_r = cv2.warpAffine(data[s],matrix,(width,height))
		label_r = cv2.warpAffine(labels[s],matrix,(width,height))
		if len(position_matricies[s]) > 0:
			position_matrix = np.multiply(position_matricies[s],np.asarray([width,height]))
			position_matrix = affine(position_matrix,matrix)


		# reshape
		data_r = data_r.reshape((height,width,1))
		label_r = label_r.reshape((height,width,1))

		for r in range(region_count_per_sample):
			# Calculate start xy of region
			cell_count_r = 0
			circle_x, circle_y = get_random_circular_xy(width/2,height/2,max_r)
			start_x = int(circle_x - region_size/2)	# Start offset of region
			start_y = int(circle_y - region_size/2)  # x,y

			# Cut out region
			data_region = data_r[start_y:start_y+region_size, 
			start_x:start_x+region_size] 
			label_region = label_r[start_y:start_y+region_size, 
			start_x:start_x+region_size]

			if len(position_matricies[s]) > 0:
				border_room_px = 3
				for i in range(len(position_matrix)):
					if position_matrix[i,0] >= start_x-border_room_px and \
					position_matrix[i,0] <= start_x+region_size+border_room_px and \
					position_matrix[i,1] >= start_y-border_room_px and \
					position_matrix[i,1] <= start_y+region_size+border_room_px:
						cell_count_r+=1


			idx = s*region_count_per_sample+r
			data_regions[idx] = data_region
			label_regions[idx] = label_region
			cell_counts[idx] = cell_count_r

	return data_regions,label_regions, cell_counts


def get_all_regions(data,labels,position_matricies,region_size):
	width,height,channels = get_dims_hwc(data[0])

	sample_rows = int(math.floor(float(height)/float(region_size)))
	sample_cols = int(math.floor(float(width)/float(region_size)))
	regions_per_sample = sample_rows*sample_cols

	data_r = np.zeros((regions_per_sample*len(data),region_size,region_size,channels),dtype=np.float64)
	label_r = np.zeros((regions_per_sample*len(labels),region_size,region_size,1),dtype=np.float64)
	cell_count_r = np.zeros([regions_per_sample*len(labels),1],dtype=np.float32)

	for i in range(len(data)):
		if len(position_matricies[i]) > 0:
			position_matrix = np.multiply(position_matricies[i],np.asarray([width,height]))

		for y in range(sample_rows):
			for x in range(sample_cols):
				c_data_r = data[i][y*region_size:y*region_size+region_size, x*region_size:x*region_size+region_size]
				c_label_r = labels[i][y*region_size:y*region_size+region_size, x*region_size:x*region_size+region_size]
				c_cell_count_r = np.asarray([0])

				if len(position_matricies[i]) > 0:
					border_room_px = 0
					for j in range(len(position_matrix)):
						if position_matrix[j,0] >= x*region_size-border_room_px and \
						position_matrix[j,0] <= x*region_size+region_size+border_room_px and \
						position_matrix[j,1] >= y*region_size-border_room_px and \
						position_matrix[j,1] <= y*region_size+region_size+border_room_px:
							c_cell_count_r[0] +=1


				idx = i*regions_per_sample+y*sample_cols+x
				data_r[idx] = c_data_r
				label_r[idx] = c_label_r
				cell_count_r[idx] = c_cell_count_r
	return data_r,label_r, cell_count_r

def as_numpy_array_of_shape(data,labels,width,height,channels):
	data_arr = np.zeros([len(data),width,height,channels],dtype=np.uint8)
	label_arr = np.zeros([len(labels),width,height,channels],dtype=np.uint8)

	for i in range(len(data)):
		d,l = data[i].copy(),labels[i].copy()

		# Resize
		if d.shape[0] != height or d.shape[1] != width:
			d = cv2.resize(d,(width,height),interpolation=cv2.INTER_CUBIC)
		if l.shape[0] != height or l.shape[1] != width:
			l = cv2.resize(l,(width,height),interpolation=cv2.INTER_CUBIC)

		# Grayscale
		d=adjust_channels(d,channels)
		l=adjust_channels(l,channels)

		data_arr[i] = d
		label_arr[i] = l

	return data_arr,label_arr


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

def adjust_channels(im,channels):	
	# Gray => Color
	if (len(im.shape) == 2 or im.shape[-1] == 1) and channels == 3:
		im = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
	# Gray [y,x] => Gray[y,x,1]
	elif len(im.shape) == 2 and channels == 1:
		im = im.reshape([im.shape[0],im.shape[1],channels])
	# Color => Gray
	elif len(im.shape) == 3 and channels == 1:
		im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).reshape([im.shape[0],im.shape[1],channels])

	return im


def affine(points, matrix):
	return np.dot(np.c_[points, np.ones(points.shape[0])], matrix.T)

def show_image_hwc(image,win_name="image",width=600,height=600,wait_key=True, destroy_window=True):	
	cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
	cv2.resizeWindow(win_name, width,height)
	cv2.imshow(win_name,image)

	if wait_key: cv2.waitKey(0)
	if destroy_window: cv2.destroyAllWindows()

def get_dims_hwc(image):
	width = image.shape[1]
	height = image.shape[0]
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

def normalize_uint8(data,labels):
	data = (data/256.0).astype(np.float32)
	labels = (labels/256.0).astype(np.float32)
	return data,labels

def denormalize_float32(data,labels):
	data = (np.clip(data,0,255)*256.0).astype(np.uint8)
	labels = (np.clip(labels,0,255)*256.0).astype(np.uint8)
	return data,labels