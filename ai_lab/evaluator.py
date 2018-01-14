import imp

from ai_lab import storagemanager as sm

'''
	load_evaluator(path[string])

	Description:
		Returns the evaluator class, based on the 
		BaseEvaluator (EvaluatorBaseStructure). It should be
		abstracted from BaseEvaluator, else NotImplemented-
		Error is raised.

	evaluator(class):
		|_ class_name = evaluator
		|_ abstracted from BaseEvaluator


'''
def load_evaluator(eval_path):
	eval_path = sm.check_file_path(eval_path)
	if eval_path == None:
		return None

	#if eval_path[-3:] == ".py":
	#	eval_path = eval_path[:-3]

	evaluator = imp.load_source("evaluator",eval_path).evaluator(eval_path)

	return evaluator


''' 
	BaseEvaluator(class)

	Description:
		Container for EvaluatorBaseStructure, holds base functionality 
		(like abstract class) of the standard evaluator.

'''
class BaseEvaluator:

	'''
	Procedure:
		-__init__

		-check_compatibility
		-evaluate_model

	'''

	# Model Parameters
	eval_path = None
	compatibilities = []
	hyper_params = None
	hyper_param_dict = None

	# Ret: evaluation string
	# class-vars:	-
	def evaluate_model(self,model):
		raise NotImplementedError()

	# Ret: Is model compatible with evaluator (boolean)
	# class-vars:	compatibilities
	def check_compatibility(self,model):
		model_comp = model.model_compatiblity
		return model_comp in self.compatibilities
		
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
