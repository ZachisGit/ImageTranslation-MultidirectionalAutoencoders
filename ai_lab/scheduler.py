from ai_lab import storagemanager as sm
import os

# Loads the scheduler.config and returns it in a dictionary
def read_scheduler_config(path):
	info_dict = {}
	with open(path,"r") as scheduler_config:
		for l in scheduler_config.readlines():
			split = l.replace("\r","").replace("\n","").split('=')
			info_dict[split[0]] = split[1]

	# Check if all necessary entries are there
	necessary_entries = ["modules_path","experiment_run","experiment_done"]
	for i in necessary_entries:
		if i not in info_dict:
			return None

	# Check if directories and files exist
	path_dir = os.path.dirname(path)+"/"
	info_dict["modules_path"] = sm.check_file_path(info_dict["modules_path"])
	info_dict["experiment_run"] = sm.check_file_path(path_dir+info_dict["experiment_run"])
	info_dict["experiment_done"] = sm.check_dir_path(path_dir+info_dict["experiment_done"])

	for k,v in info_dict.iteritems():
		if v == None:
			return None

	return info_dict

class Experiment():

	exp_data = None
	model_modules_path = None
	evaluator_modules_path = None

	# If false, discard of the experiment
	runnable = True

	# exp_data = *.exp file content
	def __init__(exp_data,model_modules_path,evaluator_modules_path):
		self.exp_data = exp_data
		self.model_modules_path = model_modules_path
		self.evaluator_modules_path = evaluator_modules_path


		if not sm.dir_exists(model_modules_path) or not sm.dir_exists(evaluator_modules_path):
			self.runnable = False
			return
