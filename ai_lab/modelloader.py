import numpy as numpy
import imp

from ai_lab import storagemanager as sm

'''
	load_model(path[string])

	Description:
		Returns the loaded model class, based on the 
		BaseModel (ModelBaseStructure). It should be
		abstracted from BaseModel, else NotImplemented-
		Error is raised.

	model(class):
		|_ class_name = model
		|_ abstracted from BaseModel


'''
def load_model(model_path):
	model_path = sm.check_file_path(model_path)
	if model_path == None:
		return None

	#if model_path[-3:] == ".py":
	#	model_path = model_path[:-3]

	model = imp.load_source("model",model_path).model(model_path)

	return model


''' 
	BaseModel(class)

	Description:
		Container for ModelBaseStructure, holds base functionality 
		(like abstract class) of the standard model.

'''
class BaseModel:

	'''
	Procedure:
		-__init__
		
		-assign_dataset
		-set_hyper_params

		-init

		-get_...
	'''

	# Model Parameters
	hyper_params = None
	hyper_param_dict = None
	dataset = None

	# Model Infos
	model_path = None
	model_name = None
	model_group = None
	model_compatibility = None

	# Assigns a dataset if dataset_loaded = True
	# else sets self.dataset to None
	# Ret: -
	# class-vars:	dataset
	def assign_dataset(self, dataset):
		if dataset.dataset_loaded == True:
			self.dataset = dataset
		else:
			self.dataset = None


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

	# Ret: -
	# class-vars:	model (tflearn model),etc
	def init(self):
		raise NotImplementedError()

	# Ret: ds.get_batch(batch_size,validation=validation)
	# class-vars:	dataset
	def get_training_batch(self,validation=False):
		raise NotImplementedError()
	
	# Ret: test_data,test_labels,cell_counts
	# class-vars:	dataset
	def get_evaluation_batch(self):
		raise NotImplementedError()

	# Ret: data,labels
	# class-vars:	dataset
	def get_livedisplay_batch(self):
		raise NotImplementedError()

	# Ret: tflearn model
	# class-vars:	model (tflearn model)
	def get_model(self):
		raise NotImplementedError()