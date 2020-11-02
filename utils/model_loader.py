import pickle as pkl
from bvcopula import IndependenceCopula_Likelihood

def get_likelihoods(summary_path,n1,n2=None):
	'''
	Looks up the likelihoods of the best selected model in summary.pkl
	Parameters
	----------
	summary_path: str
		A path to a summary of model selection
	n1: int
		The number of the first variable
	n2: int (optional)
		The number of the second variable
		If not given: assume C-vine and n1>0
	Returns
	-------
	likelihoods: list
		A list of copula likelihood objects, corresponding to the
		best selected model for a given pair of variables.
	'''
	if n2 is None:
		assert n1>0
		with open(summary_path,'rb') as f:	
			data = pkl.load(f)	
		if data[n] is not None:
			return data[n][0]
		else:
			return [IndependenceCopula_Likelihood()]
	else:
		with open(summary_path,'rb') as f:
			data = pkl.load(f)	
		if data[n1+5,n2+5] is not None:
			return data[n1+5,n2+5][0]
		else:
			return [IndependenceCopula_Likelihood()]

def get_model(weights_file,likelihoods,device):
	'''
	Loads the weights of the best selected model and returns
	the bvcopula.MultitaskGPModel object
	Parameters
	----------
	weights_file: str
		A path to the folder, containing the results of the model selection

	'''
	import glob
	from bvcopula import load_model
	try:
		model = load_model(weights_file, likelihoods, device)
		return model
	except FileNotFoundError:
		print(f'Weights file {weights_file} not found.')
		return 0