from ai_lab import storagemanager as sm

import tflearn
import time

hyper_param_dict = {
	"n_epoch": 1,
	"run_id": "run_id_x",
	"show_metric": True,
	"batch_size": 64
	} 


escape_condition_dict = {
	"time_sec": 1000, 
	"loss_smaller_then": 0.01, 
	"accuracy_bigger_then": 0.97,
	"epochs": 100,
	"training_steps": 10000
}

# To be used for scheudler key interrupts and stuff like that.
# It saves the model and exits correctly. (Not Immediate)
STOP_TRAINING = False

def train_model(model, escape_condition,model_path, hyper_params={}):
	STOP_TRAINING = False

	# Get hyper parameters
	hp = _set_hyper_params(hyper_params)
	mc = MonitorCallback()

	start_time = time.time()

	while True:
		train_data, train_labels = model.get_training_batch(validation=False)
		test_data, test_labels = model.get_training_batch(validation=True)

		model.model.fit({"input": train_data}, {"target": train_labels},
			n_epoch=hp["n_epoch"], run_id=hp["run_id"],
			validation_set=({"input": test_data}, {"target": test_labels}),
			show_metric=hp["show_metric"],callbacks=mc)

		model.model.save(model_path)

		# True => done training
		if _check_escape_condition(escape_condition, start_time, mc) or STOP_TRAINING:
			break

	train_info_dict = {
	"epochs":mc.training_state.epoch,
	"training_steps":mc.training_state.step,
	"loss":mc.training_state.global_loss,
	"accuracy":mc.training_state.global_acc,
	"best_accuracy":mc.training_state.best_accuracy,
	"time":time.time()-start_time
	}

	return model, train_info_dict



'''
	Set the hyper-parameters based on the hyper_param_dict
	if the dict is None raise NotImplementedError
	just igonre hyper_parameters not set in hyper_param_dict
	
	class-vars:	hyper_params, hyper_param_dict
	Ret: -
'''
def _set_hyper_params(hyper_params):
	global hyper_param_dict
	if hyper_param_dict == None:
		raise NotImplementedError()

	# Set all parameters defined in hyper_params,
	# for the rest use the predefined values in
	# global hyper_param_dict.
	n_hyper_params = {}
	for key,value in hyper_param_dict.iteritems():
		if key in hyper_params:
			n_hyper_params[key] = hyper_params[key]
		else:
			n_hyper_params[key] = value

	return n_hyper_params

# True = EC met so stop
# False = EC not met so continue
def _check_escape_condition(ec, start_time, monitor_callback):
	global escape_condition_dict
	if not ec.ec_name in escape_condition_dict:
		return True

	escape = False

	# Time_SEC
	if ec.ec_name == "time_sec":
		now = time.time()
		if now >= start_time+ec.ec_value:
			escape = True
	# Loss Smaller Then
	elif ec.ec_name == "loss_smaller_then":
		if monitor_callback.training_state.global_loss <= ec.ec_value:
			escape = True
	# Accuracy Bigger Then
	elif ec.ec_name == "accuracy_bigger_then":
		if monitor_callback.training_state.global_acc >= ec.ec_value:
			escape = True
	# Epochs 
	elif ec.ec_name == "epochs":
		if monitor_callback.training_state.epoch >= ec.ec_value:
			escape = True
	# Training Steps
	elif ec.ec_name == "training_steps":
		if monitor_callback.training_state.step >= ec.ec_value:
			escape = True
	else:
		escape = True

	return escape




'''
MonitorCallback
   |_training_state:
		epoch = 0
	    step = 0
	    current_iter = 0
	    step_time = 0.0

	    acc_value = None
	    loss_value = None

	    val_acc = None
	    val_loss = None

	    best_accuracy = 0.0

	    global_acc = 0.0
	    global_loss = 0.0
'''
class MonitorCallback(tflearn.callbacks.Callback):
	def __init__(self):
		self.training_state = None

	def on_epoch_end(self, training_state):
		self.training_state = training_state


''' 
EscapeCondition:
	|_ec_name
	|_ec_value

	escape_condition_dict = {
		"time_sec": 1000, 
		"loss_smaller_then": 0.01, 
		"accuracy_bigger_then": 0.97,
		"epochs": 100,
		"training_steps": 10000
	}
'''
class EscapeCondition:

	ec_name = None
	ec_value = None

	def __init__(self, ec_name, ec_value):
		self.ec_name = ec_name
		self.ec_value = ec_value