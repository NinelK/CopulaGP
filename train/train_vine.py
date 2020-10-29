from utils import standard_loader
from train import train_next_tree
from train import save_checkpoint, load_checkpoint, save_final
from typing import Callable

def train_vine(exp: str, path_data: Callable[[int],str], 
		path_models: Callable[[int],str], path_final: str,
		layers_max=-1,start=0,gauss=False,device_list=['cpu']):
	'''
	Trains a vine model layer by layer, saving
	the checkpoints between the layers
	Parameters
	----------
	exp : str
		Name of the experiment
	path_data : Callable[[int],str]
		A lambda function that takes a layer number
		and returns a path to data inputs to this layer
	path_models : Callable[[int],str]
		A lambda function that takes a layer number
		and returns a path to Pair_CopulaGP models
		for this layer
	path_final : str
		Where to store the final results
	layers_max : int (Default = -1)
		Maximal numper of vine trees to train.
		If -1 : train all (N-1 trees).
	start : int (Default = 0)
		The first vine tree to start from.
		Can be used to resume training from 
		a checkpoint.
	gauss : bool (Default = False)
		A flag that turns off model selection
		and only trains gaussian copula models
	device_list : List[str] (Default = ['cpu'])
		A list of devices to be used for
		training (in parallel)

	Returns
	-------
	to_save : dict
		Dictionary with keys={'models','waics'}
	'''

	X,Y = standard_loader(path_data(0))

	# figure out how many trees to train
	layers = Y.shape[-1]-1 if layers_max == -1 else layers_max

	if start == 0:
		to_save = {}
		to_save['models'], to_save['waics'] = [],[]
	else:
		X,Y,to_save = load_checkpoint(path_data(start),path_models(start-1))
	for layer in range(start,layers):
		print(f'Starting {exp} layer {layer}/{layers}')
		model, waic, Y = train_next_tree(X,Y,layer,device_list,gauss=gauss,exp=exp)
		to_save['models'].append(model)
		to_save['waics'].append(waic)
		# save checkpoint
		save_checkpoint(X,Y,to_save,path_data(layer+1),path_models(layer))

	save_final(path_data(0),path_models(layers-2),path_final)

	return to_save

def train_vine_light(X,Y,layers_max=-1,gauss=False,device_list=['cpu']):
	'''
	Same as train_vine, but does not
	save any files. Takes (X,Y) as an input
	and outputs a trained model.
	Suitable for small models.

	Parameters
	----------
	X : np.ndarray
		Conditioning variable
	Y : np.ndarray
		Collection of data variables
	layers_max : int (Default = -1)
		Maximal numper of vine trees to train.
		If -1 : train all (N-1 trees).
	gauss : bool (Default = False)
		A flag that turns off model selection
		and only trains gaussian copula models
	device_list : List[str] (Default = ['cpu'])
		A list of devices to be used for
		training (in parallel)

	Returns
	-------
	to_save : dict
		Dictionary with keys={'models','waics'}
	'''

	assert X.shape == Y[:,0].shape

	# figure out how many trees to train
	layers = Y.shape[-1]-1 if layers_max == -1 else layers_max

	to_save = {}
	to_save['models'], to_save['waics'] = [],[]

	for layer in range(layers):
		print(f'Starting layer {layer}/{layers}')
		model, waic, Y = train_next_tree(X,Y,layer,device_list,gauss=gauss,exp='')
		to_save['models'].append(model)
		to_save['waics'].append(waic)

	return to_save