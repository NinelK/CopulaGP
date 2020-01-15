import pickle as pkl
import numpy as np

def load_experimental_data(path,animal,day_name,n1,n2):
	'''
		Loads experimental data
		TODO: squash parameters into "{}/{}_{}".format(path,animal,day_name)
	'''
	def data_from_n(n):
		if n>=0:
			data = signals[n]
		elif n==-1:
			data = behaviour_pkl['transformed_velocity']
		elif n==-2:
			data = behaviour_pkl['transformed_licks']
		elif n==-3:
			data = (behaviour_pkl['transformed_early_reward'] + behaviour_pkl['transformed_late_reward'])/2
		elif n==-4:
			data = behaviour_pkl['transformed_early_reward']
		elif n==-5:
			data = behaviour_pkl['transformed_late_reward']
		else:
			raise ValueError('n is out of range')
		return data

	with open("{}/{}_{}_signals.pkl".format(path,animal,day_name),'rb') as f:
	    signal_pkl = pkl.load(f)
	with open("{}/{}_{}_behaviour.pkl".format(path,animal,day_name),'rb') as f:
	    behaviour_pkl = pkl.load(f)
	for s in ['ROIsN','trialStart','maxTrialNum','trials']:
	    assert(np.allclose(signal_pkl[s],behaviour_pkl[s]))

	signals = signal_pkl['signals_transformed']

	data1 = data_from_n(n1)
	data2 = data_from_n(n2)

	Y_all = np.array([data1,data2]).T
	X_all = np.array(behaviour_pkl['position'])#local_time

	rule = (Y_all[:,0]>0) & (Y_all[:,1]>0)  \
	        & (Y_all[:,0]<1) & (Y_all[:,1]<1)
	 
	X = np.reshape(X_all[rule],(-1,1))
	X[X<0] = 160.+X[X<0]
	X[X>160] = X[X>160]-160.
	X = X/160.
	Y = Y_all[rule]
	
	return X, Y

def get_likelihoods(summary_path,n1,n2):
	'''
	Looks up the likelihoods of the best selected model in summary.pkl
	Parameters
	----------
	summary_path: str
		A path to a summary of model selection
	n1: int
		The number of the first variable
	n2: int
		The number of the second variable
	Returns
	-------
	likelihoods: list
		A list of copula likelihood objects, corresponding to the
		best selected model for a given pair of variables.
	'''
	with open(summary_path,'rb') as f:
		data = pkl.load(f)	
	return data[n1+5,n2+5][0]

def get_model(weights_file,likelihoods,device):
	'''
	Loads the weights of the best selected model and returns
	the bvcopula.Mixed_GPInferenceModel object
	Parameters
	----------
	weights_file: str
		A path to the folder, containing the results of the model selection

	'''
	import glob
	from bvcopula import load_model
	get_weights_filename = glob.glob(weights_file)
	print(get_weights_filename)
	if len(get_weights_filename)>0:
		if len(get_weights_filename)>1:
			print('There is more then 1 file, taking the first one')
			return 0
		model = load_model(get_weights_filename[0], likelihoods, device)
		return model
	else:
		print('Weights file {} not found.'.format(get_weights_filename))
		return 0